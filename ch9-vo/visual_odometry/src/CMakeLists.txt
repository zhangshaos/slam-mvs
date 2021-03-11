find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_path(SOPHUS_INCLUDE_DIRS "sophus/common.hpp")

include_directories(${SOPHUS_INCLUDE_DIRS})

add_library(vo SHARED
    frame.cpp
    mappoint.cpp
    map.cpp
    camera.cpp
    config.cpp
    g2o_types.cpp
    visual_odometry.cpp
)

target_link_libraries(vo PRIVATE Eigen3::Eigen ${OpenCV_LIBS} g2o::core g2o::stuff g2o::types_icp g2o::types_sba)