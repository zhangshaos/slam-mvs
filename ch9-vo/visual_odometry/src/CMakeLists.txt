aux_source_directory(. SRC)

add_library(vo STATIC ${SRC})

target_link_libraries(vo PRIVATE Eigen3::Eigen ${OpenCV_LIBS} g2o::core g2o::stuff g2o::types_icp g2o::types_sba)