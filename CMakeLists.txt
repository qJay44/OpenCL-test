include(./.env.cmake OPTIONAL RESULT_VARIABLE LOCAL_ENV)
message(STATUS "Local .env.cmake: ${LOCAL_ENV}")

cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER ${MINGW32_PATH}gcc.exe)
set(CMAKE_CXX_COMPILER ${MINGW32_PATH}g++.exe)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(MyProject VERSION 1.0)

include_directories(${MY_UTILS_PATH})

add_executable(MyProject
  ${SRC_DIR}main.cpp
)

target_include_directories(MyProject PUBLIC ${OPENCL_PATH}/include)
target_link_directories(MyProject PUBLIC ${OPENCL_PATH}/lib/x86)
target_link_libraries(MyProject PUBLIC OpenCL)

