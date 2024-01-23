cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(simple_share_tensor LANGUAGES C CXX CUDA)

find_package(CUDA 10.2 REQUIRED)

option(AWS_BUILD "Build projects on AWS" OFF)
option(AWS_SDK_PATH "AWS SDK PATH" OFF)

message("-- AWS_SDK_PATH: ${AWS_SDK_PATH}")
if (AWS_SDK_PATH)
  add_definitions("-DAWS_SDK_ON")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
# set(CMAKE_CXX_STANDARD 14)

add_definitions("-DBUILD_MULTI_GPU")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
find_package(MPI REQUIRED)
find_package(NCCL REQUIRED)
set(CMAKE_MODULE_PATH "") # prevent the bugs for pytorch building

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
list(APPEND CMAKE_MODULE_PATH ${CUDA_PATH}/lib64)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
)


message("-- COMMON_HEADER_DIRS: ${COMMON_HEADER_DIRS}")

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib64
)

if(AWS_BUILD)
  list(APPEND COMMON_LIB_DIRS ${CUDA_PATH}/lib)
endif()

list(APPEND COMMON_HEADER_DIRS ${MPI_INCLUDE_PATH})
list(APPEND COMMON_LIB_DIRS /usr/local/mpi/lib)

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

if (AWS_SDK_PATH)
  include_directories(
    ${AWS_SDK_PATH}/include
  )
  link_directories(
    ${AWS_SDK_PATH}/lib
  )
endif()

add_subdirectory(3rdparty)
add_subdirectory(src)
add_subdirectory(test)

add_executable(simple_share simple_tensor.h simple_tensor.cu simple_share.cc)
target_link_libraries(simple_share PUBLIC matrix_transpose_kernels)
