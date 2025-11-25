cmake_minimum_required(VERSION 2.8.3)
project(vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

# ============ CUDA 配置 ============
# 设置 CUDA 路径
set(CUDA_ROOT "/usr/local/cuda-11.8")

# 查找 CUDA 运行时库
find_library(CUDA_CUDART_LIBRARY
    NAMES cudart
    PATHS ${CUDA_ROOT}/lib64
          /usr/local/cuda/lib64
    NO_DEFAULT_PATH
)

# 查找 CUDA 其他必要库
find_library(CUDA_CUDA_LIBRARY
    NAMES cuda
    PATHS ${CUDA_ROOT}/lib64/stubs
    NO_DEFAULT_PATH
)

# 检查 CUDA 库是否找到
if(CUDA_CUDART_LIBRARY)
    message(STATUS "Found CUDA runtime: ${CUDA_CUDART_LIBRARY}")
    set(CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY})
    
    # 如果找到 cuda stubs 库，也添加
    if(CUDA_CUDA_LIBRARY)
        list(APPEND CUDA_LIBRARIES ${CUDA_CUDA_LIBRARY})
    endif()
else()
    message(WARNING "CUDA runtime library not found, trying system search")
    # 让链接器尝试系统路径查找
    set(CUDA_LIBRARIES cudart)
endif()

# 添加 CUDA 头文件路径
find_path(CUDA_INCLUDE_DIR
    NAMES cuda_runtime.h
    PATHS ${CUDA_ROOT}/include
    NO_DEFAULT_PATH
)

if(CUDA_INCLUDE_DIR)
    include_directories(${CUDA_INCLUDE_DIR})
    message(STATUS "Found CUDA includes: ${CUDA_INCLUDE_DIR}")
endif()

# ============ TensorRT 配置 ============
set(TENSORRT_ROOT "/opt/tensorrt")

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${TENSORRT_ROOT}/include
    NO_DEFAULT_PATH
)

find_library(TENSORRT_LIBRARY
    NAMES nvinfer
    PATHS ${TENSORRT_ROOT}/lib
    NO_DEFAULT_PATH
)

find_library(TENSORRT_ONNXPARSER_LIBRARY
    NAMES nvonnxparser
    PATHS ${TENSORRT_ROOT}/lib
    NO_DEFAULT_PATH
)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    message(STATUS "Found TensorRT: ${TENSORRT_INCLUDE_DIR}")
    include_directories(${TENSORRT_INCLUDE_DIR})
    add_definitions(-DUSE_TENSORRT)
else()
    message(WARNING "TensorRT not found, building without TensorRT support")
endif()

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIR}
)

catkin_package()

add_library(vins_lib
    src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp
    src/featureTracker/sparse_flow.cpp)

# 修改链接库配置
if(TENSORRT_LIBRARY AND TENSORRT_ONNXPARSER_LIBRARY)
    target_link_libraries(vins_lib 
        ${catkin_LIBRARIES} 
        ${OpenCV_LIBS} 
        ${CERES_LIBRARIES} 
        ${CUDA_LIBRARIES}  # 使用我们找到的 CUDA 库
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNXPARSER_LIBRARY})
else()
    target_link_libraries(vins_lib 
        ${catkin_LIBRARIES} 
        ${OpenCV_LIBS} 
        ${CERES_LIBRARIES} 
        ${CUDA_LIBRARIES})  # 使用我们找到的 CUDA 库
endif()

add_executable(vins_node src/rosNodeTest.cpp)
target_link_libraries(vins_node vins_lib) 

add_executable(kitti_odom_test src/KITTIOdomTest.cpp)
target_link_libraries(kitti_odom_test vins_lib) 

add_executable(kitti_gps_test src/KITTIGPSTest.cpp)
target_link_libraries(kitti_gps_test vins_lib) 

add_executable(uma_node src/umaviTest.cpp)
target_link_libraries(uma_node vins_lib)

add_executable(sea_node src/deepseaTest.cpp)
target_link_libraries(sea_node vins_lib)