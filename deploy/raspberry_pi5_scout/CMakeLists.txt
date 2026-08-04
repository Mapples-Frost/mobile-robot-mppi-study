cmake_minimum_required(VERSION 3.16)
project(rlmppi_livox_udp_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIVOX_SDK_ROOT "/home/pi/Livox-SDK2-master" CACHE PATH
    "Path to the compiled Livox-SDK2 source tree")

add_executable(rlmppi_livox_udp_bridge livox_udp_bridge.cpp)
target_include_directories(rlmppi_livox_udp_bridge PRIVATE
    "${LIVOX_SDK_ROOT}/include")
target_link_libraries(rlmppi_livox_udp_bridge PRIVATE
    "${LIVOX_SDK_ROOT}/build/sdk_core/liblivox_lidar_sdk_static.a"
    pthread)
target_compile_options(rlmppi_livox_udp_bridge PRIVATE -O3 -Wall -Wextra)
