cmake_minimum_required(VERSION 2.8.3)
project(corner_tracking_trt)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# ============ OpenCV 配置 ============
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

# ============ CUDA 配置 ============
# 设置 CUDA 路径
set(CUDA_ROOT "/usr/local/cuda-11.8" CACHE PATH "CUDA installation directory")

# 查找 CUDA 运行时库
find_library(CUDA_CUDART_LIBRARY
    NAMES cudart
    PATHS ${CUDA_ROOT}/lib64
          /usr/local/cuda/lib64
          /usr/local/cuda-11.8/lib64
    NO_DEFAULT_PATH
)

# 查找 CUDA 其他必要库
find_library(CUDA_CUDA_LIBRARY
    NAMES cuda
    PATHS ${CUDA_ROOT}/lib64/stubs
          /usr/local/cuda/lib64/stubs
    NO_DEFAULT_PATH
)

# 检查 CUDA 库是否找到
if(CUDA_CUDART_LIBRARY)
    message(STATUS "Found CUDA runtime: ${CUDA_CUDART_LIBRARY}")
    set(CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY})
    
    # 如果找到 cuda stubs 库，也添加
    if(CUDA_CUDA_LIBRARY)
        list(APPEND CUDA_LIBRARIES ${CUDA_CUDA_LIBRARY})
        message(STATUS "Found CUDA driver stub:  ${CUDA_CUDA_LIBRARY}")
    endif()
else()
    message(WARNING "CUDA runtime library not found, trying system search")
    set(CUDA_LIBRARIES cudart)
endif()

# 添加 CUDA 头文件路径
find_path(CUDA_INCLUDE_DIR
    NAMES cuda_runtime.h
    PATHS ${CUDA_ROOT}/include
          /usr/local/cuda/include
          /usr/local/cuda-11.8/include
    NO_DEFAULT_PATH
)

if(CUDA_INCLUDE_DIR)
    include_directories(${CUDA_INCLUDE_DIR})
    message(STATUS "Found CUDA includes: ${CUDA_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "CUDA include directory not found!")
endif()

# ============ TensorRT 配置 ============
set(TENSORRT_ROOT "/opt/tensorrt" CACHE PATH "TensorRT installation directory")

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${TENSORRT_ROOT}/include
          /usr/include/x86_64-linux-gnu
          /usr/local/include
    NO_DEFAULT_PATH
)

find_library(TENSORRT_LIBRARY
    NAMES nvinfer
    PATHS ${TENSORRT_ROOT}/lib
          /usr/lib/x86_64-linux-gnu
          /usr/local/lib
    NO_DEFAULT_PATH
)

find_library(TENSORRT_ONNXPARSER_LIBRARY
    NAMES nvonnxparser
    PATHS ${TENSORRT_ROOT}/lib
          /usr/lib/x86_64-linux-gnu
          /usr/local/lib
    NO_DEFAULT_PATH
)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
    message(STATUS "Found TensorRT include: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "Found TensorRT library: ${TENSORRT_LIBRARY}")
    include_directories(${TENSORRT_INCLUDE_DIR})
    add_definitions(-DUSE_TENSORRT)
    set(TENSORRT_FOUND TRUE)
else()
    message(FATAL_ERROR "TensorRT not found!  Please set TENSORRT_ROOT correctly.")
    set(TENSORRT_FOUND FALSE)
endif()

# ============ 头文件目录 ============
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
)

# ============ 源文件 ============
set(TRACKING_SOURCES
    tracking_trt.cpp
)

set(TRACKING_HEADERS
    tracking_trt.h
)

# ============ 构建库 ============
add_library(tracking_trt_lib ${TRACKING_SOURCES})

# 链接库
if(TENSORRT_FOUND)
    target_link_libraries(tracking_trt_lib
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${TENSORRT_LIBRARY}
        ${TENSORRT_ONNXPARSER_LIBRARY}
        pthread
    )
else()
    target_link_libraries(tracking_trt_lib
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        pthread
    )
endif()

# ============ 构建可执行文件 ============
add_executable(demo_trt main.cpp)
target_link_libraries(demo_trt tracking_trt_lib)

# ============ 安装规则 ============
install(TARGETS demo_trt tracking_trt_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${TRACKING_HEADERS}
    DESTINATION include/corner_tracking_trt
)

# ============ 打印配置信息 ============
message(STATUS "================================")
message(STATUS "Build Configuration Summary:")
message(STATUS "================================")
message(STATUS "Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Flags:        ${CMAKE_CXX_FLAGS}")
message(STATUS "OpenCV Version:   ${OpenCV_VERSION}")
message(STATUS "CUDA Root:        ${CUDA_ROOT}")
message(STATUS "CUDA Libraries:   ${CUDA_LIBRARIES}")
message(STATUS "TensorRT Root:    ${TENSORRT_ROOT}")
message(STATUS "TensorRT Found:   ${TENSORRT_FOUND}")
message(STATUS "================================")