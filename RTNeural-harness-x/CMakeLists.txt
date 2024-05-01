cmake_minimum_required(VERSION 3.5)
project(RTNeural-harness-x)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

set(RTNEURAL_STL ON CACHE BOOL "Use RTNeural with this backend" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../RTNeural-x" modules/RTNeural)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../RTNeural-x/RTNeural")
target_compile_features(RTNeural PRIVATE cxx_std_20)

add_compile_definitions(
        RTNEURAL_USE_APPROX=1
        RTNEURAL_STL_PERF_CNN=1
        # RTNEURAL_USE_RAND_MATMUL=1
)

add_executable(harness
        src/harness.cpp
        src/dr_wav.c)
target_link_libraries(harness PUBLIC RTNeural)
