cmake_minimum_required(VERSION 3.18)
project(nearfield360_deploy VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler optimization flags
if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# Header-only core interface library
add_library(nearfield360_cpp INTERFACE)
target_include_directories(nearfield360_cpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Optional TensorRT and CUDA integration
find_package(CUDA QUIET)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/x86_64-linux-gnu /usr/local/cuda/include $ENV{TENSORRT_PATH}/include
)

if(CUDA_FOUND AND TENSORRT_INCLUDE_DIR)
    message(STATUS "Native CUDA and TensorRT SDK detected. Enabling hardware acceleration.")
    target_compile_definitions(nearfield360_cpp INTERFACE NEARFIELD360_HAS_TENSORRT=1)
    target_include_directories(nearfield360_cpp INTERFACE ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
else()
    message(STATUS "Native CUDA/TensorRT SDK not found in environment. Building in standalone deployment wrapper mode.")
endif()

# Standalone benchmark deployment executable
add_executable(nearfield360_benchmark src/main.cpp)
target_link_libraries(nearfield360_benchmark PRIVATE nearfield360_cpp)

# Installation targets
install(TARGETS nearfield360_benchmark nearfield360_cpp
    EXPORT nearfield360_deploy_targets
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
