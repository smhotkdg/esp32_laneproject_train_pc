#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "camera_pins.h"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "esp_camera.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

static const char *TAG = "lane_s3";

static constexpr int MODEL_W = 160;
static constexpr int MODEL_H = 96;
static constexpr float ROI_TOP_RATIO = 0.375f;
static constexpr int CAMERA_W = 320;
static constexpr int CAMERA_H = 240;

struct LaneResult {
    bool has_left = false;
    bool has_right = false;
    int left_x = -1;
    int right_x = -1;
    int lane_center_x = -1;
    int frame_center_x = MODEL_W / 2;
    int offset_x = 0;
    int positive_pixels = 0;
};

static camera_config_t camera_config = {
    .pin_pwdn = CAM_PIN_PWDN,
    .pin_reset = CAM_PIN_RESET,
    .pin_xclk = CAM_PIN_XCLK,
    .pin_sccb_sda = CAM_PIN_SIOD,
    .pin_sccb_scl = CAM_PIN_SIOC,
    .pin_d7 = CAM_PIN_D7,
    .pin_d6 = CAM_PIN_D6,
    .pin_d5 = CAM_PIN_D5,
    .pin_d4 = CAM_PIN_D4,
    .pin_d3 = CAM_PIN_D3,
    .pin_d2 = CAM_PIN_D2,
    .pin_d1 = CAM_PIN_D1,
    .pin_d0 = CAM_PIN_D0,
    .pin_vsync = CAM_PIN_VSYNC,
    .pin_href = CAM_PIN_HREF,
    .pin_pclk = CAM_PIN_PCLK,
    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_GRAYSCALE,
    .frame_size = FRAMESIZE_QVGA,
    .jpeg_quality = 12,
    .fb_count = 1,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

static inline uint8_t sample_bilinear_gray(
    const uint8_t *src,
    int src_w,
    int src_h,
    float x,
    float y)
{
    x = std::max(0.0f, std::min(x, static_cast<float>(src_w - 1)));
    y = std::max(0.0f, std::min(y, static_cast<float>(src_h - 1)));

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = std::min(x0 + 1, src_w - 1);
    int y1 = std::min(y0 + 1, src_h - 1);

    float wx = x - x0;
    float wy = y - y0;

    float v00 = src[y0 * src_w + x0];
    float v01 = src[y0 * src_w + x1];
    float v10 = src[y1 * src_w + x0];
    float v11 = src[y1 * src_w + x1];

    float top = v00 + (v01 - v00) * wx;
    float bottom = v10 + (v11 - v10) * wx;
    float value = top + (bottom - top) * wy;

    value = std::max(0.0f, std::min(value, 255.0f));
    return static_cast<uint8_t>(value + 0.5f);
}

static esp_err_t init_camera()
{
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: %s", esp_err_to_name(err));
        return err;
    }
    return ESP_OK;
}

static void fill_input_tensor_from_frame(
    dl::TensorBase *input_tensor,
    const camera_fb_t *fb)
{
    const std::vector<int> shape = input_tensor->get_shape();
    if (shape.size() != 4) {
        ESP_LOGE(TAG, "Unexpected input tensor rank: %d", static_cast<int>(shape.size()));
        return;
    }

    const int roi_y0 = static_cast<int>(fb->height * ROI_TOP_RATIO);
    const int roi_h = fb->height - roi_y0;
    const int src_w = fb->width;
    const int src_h = roi_h;
    const uint8_t *src = fb->buf + roi_y0 * fb->width;

    float *dst = static_cast<float *>(input_tensor->data);
    if (dst == nullptr) {
        ESP_LOGE(TAG, "Input tensor data is null");
        return;
    }

    // Support either NCHW [1,1,H,W] or NHWC [1,H,W,1]
    bool nchw = (shape[1] == 1 && shape[2] == MODEL_H && shape[3] == MODEL_W);
    bool nhwc = (shape[1] == MODEL_H && shape[2] == MODEL_W && shape[3] == 1);

    if (!nchw && !nhwc) {
        ESP_LOGW(TAG, "Unexpected input shape: [%d, %d, %d, %d], trying NCHW fill",
                 shape[0], shape[1], shape[2], shape[3]);
        nchw = true;
    }

    for (int y = 0; y < MODEL_H; ++y) {
        float sy = (static_cast<float>(y) + 0.5f) * static_cast<float>(src_h) / static_cast<float>(MODEL_H) - 0.5f;
        for (int x = 0; x < MODEL_W; ++x) {
            float sx = (static_cast<float>(x) + 0.5f) * static_cast<float>(src_w) / static_cast<float>(MODEL_W) - 0.5f;
            const uint8_t pix = sample_bilinear_gray(src, src_w, src_h, sx, sy);
            const float value = static_cast<float>(pix) / 255.0f;

            if (nchw) {
                dst[y * MODEL_W + x] = value;
            } else if (nhwc) {
                dst[(y * MODEL_W + x)] = value;
            }
        }
    }
}

static LaneResult postprocess_lane_mask(dl::TensorBase *output_tensor)
{
    LaneResult result;
    dl::TensorBase float_mask(output_tensor->get_shape(), nullptr, 0, dl::DATA_TYPE_FLOAT);
    float_mask.assign(output_tensor);

    const std::vector<int> shape = float_mask.get_shape();
    const float *ptr = static_cast<const float *>(float_mask.data);
    if (ptr == nullptr || shape.size() != 4) {
        ESP_LOGE(TAG, "Invalid output tensor");
        return result;
    }

    bool nchw = (shape[1] == 1 && shape[2] == MODEL_H && shape[3] == MODEL_W);
    bool nhwc = (shape[1] == MODEL_H && shape[2] == MODEL_W && shape[3] == 1);

    std::vector<uint8_t> mask(MODEL_W * MODEL_H, 0);

    for (int y = 0; y < MODEL_H; ++y) {
        for (int x = 0; x < MODEL_W; ++x) {
            float v = 0.0f;
            if (nchw) {
                v = ptr[y * MODEL_W + x];
            } else if (nhwc) {
                v = ptr[y * MODEL_W + x];
            }
            uint8_t on = (v >= 0.50f) ? 1 : 0;
            mask[y * MODEL_W + x] = on;
            result.positive_pixels += static_cast<int>(on);
        }
    }

    std::vector<int> left_points;
    std::vector<int> right_points;

    for (int y = MODEL_H - 1; y >= static_cast<int>(MODEL_H * 0.20f); y -= 3) {
        int best_left = -1;
        int best_right = -1;
        for (int x = 0; x < MODEL_W; ++x) {
            if (mask[y * MODEL_W + x] == 0) {
                continue;
            }
            if (x < MODEL_W / 2) {
                best_left = x;
            } else if (best_right < 0) {
                best_right = x;
            }
        }
        if (best_left >= 0) {
            left_points.push_back(best_left);
        }
        if (best_right >= 0) {
            right_points.push_back(best_right);
        }
    }

    if (!left_points.empty()) {
        result.has_left = true;
        result.left_x = left_points.front();
    }
    if (!right_points.empty()) {
        result.has_right = true;
        result.right_x = right_points.front();
    }

    if (result.has_left && result.has_right) {
        result.lane_center_x = (result.left_x + result.right_x) / 2;
        result.offset_x = result.lane_center_x - result.frame_center_x;
    }

    return result;
}

extern "C" void app_main(void)
{
    ESP_ERROR_CHECK(init_camera());

    dl::Model *model = new dl::Model(
        reinterpret_cast<const char *>(model_espdl),
        fbs::MODEL_LOCATION_IN_FLASH_RODATA,
        128 * 1024,
        dl::MEMORY_MANAGER_GREEDY,
        nullptr,
        true);

    model->profile_memory();

    std::map<std::string, dl::TensorBase *> inputs = model->get_inputs();
    std::map<std::string, dl::TensorBase *> outputs = model->get_outputs();
    if (inputs.empty() || outputs.empty()) {
        ESP_LOGE(TAG, "Model input/output not found");
        return;
    }

    dl::TensorBase *model_input = inputs.begin()->second;
    dl::TensorBase *model_output = outputs.begin()->second;
    dl::TensorBase *float_input = new dl::TensorBase(model_input->get_shape(), nullptr, 0, dl::DATA_TYPE_FLOAT);

    ESP_LOGI(TAG, "Input shape ready.");
    ESP_LOGI(TAG, "Starting lane inference loop...");

    int64_t frame_idx = 0;
    while (true) {
        const int64_t t0 = esp_timer_get_time();
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed");
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }
        const int64_t t1 = esp_timer_get_time();

        fill_input_tensor_from_frame(float_input, fb);
        const int64_t t2 = esp_timer_get_time();

        model->run(float_input, dl::RUNTIME_MODE_AUTO);
        const int64_t t3 = esp_timer_get_time();

        LaneResult lane = postprocess_lane_mask(model_output);
        const int64_t t4 = esp_timer_get_time();

        esp_camera_fb_return(fb);

        if ((frame_idx % 30) == 0) {
            model->profile_module(true);
        }

        if (lane.has_left && lane.has_right) {
            ESP_LOGI(
                TAG,
                "[%lld] cap=%lldus prep=%lldus infer=%lldus post=%lldus total=%lldus | left=%d right=%d center=%d offset=%+d pos=%d",
                static_cast<long long>(frame_idx),
                static_cast<long long>(t1 - t0),
                static_cast<long long>(t2 - t1),
                static_cast<long long>(t3 - t2),
                static_cast<long long>(t4 - t3),
                static_cast<long long>(t4 - t0),
                lane.left_x,
                lane.right_x,
                lane.lane_center_x,
                lane.offset_x,
                lane.positive_pixels);
        } else {
            ESP_LOGW(
                TAG,
                "[%lld] cap=%lldus prep=%lldus infer=%lldus post=%lldus total=%lldus | lane not stable (pos=%d)",
                static_cast<long long>(frame_idx),
                static_cast<long long>(t1 - t0),
                static_cast<long long>(t2 - t1),
                static_cast<long long>(t3 - t2),
                static_cast<long long>(t4 - t3),
                static_cast<long long>(t4 - t0),
                lane.positive_pixels);
        }

        ++frame_idx;
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}
