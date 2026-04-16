set(srcs app_main.cpp)

set(requires esp-dl esp32-camera)

idf_build_get_property(component_targets __COMPONENT_TARGETS)

if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()

set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

set(embed_files models/model.espdl)

idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "."
    REQUIRES ${requires}
)

target_add_aligned_binary_data(${COMPONENT_LIB} "models/model.espdl" TEXT)
