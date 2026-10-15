cmake_minimum_required(VERSION 3.24)
project(Ataxx LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include_directories(../python/src)
add_executable(bot bot.cpp)
target_compile_options(bot PRIVATE -Wall -Wextra -Wpedantic)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/botzone.cpp"
    COMMAND "${CMAKE_COMMAND}" -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/botzone.cpp -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Submission.cmake
    DEPENDS ../python/src/board.h weights.h patterns.h directional.h wide.h engine.hpp bot.cpp cmake/Submission.cmake
    VERBATIM
)
add_custom_target(submission DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/botzone.cpp")

option(ATAXX_CUDA "Build the GPU arena" OFF)
if(ATAXX_CUDA)
    enable_language(CUDA)
    add_executable(arena arena.cu)
    set_target_properties(arena PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(arena PRIVATE --expt-relaxed-constexpr -diag-suppress=20014)
endif()

option(ATAXX_DESKTOP "Build the desktop game" ON)
if(ATAXX_DESKTOP)
    include(FetchContent)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(raylib
        URL https://github.com/raysan5/raylib/archive/dbc56a87da87d973a9c5baa4e7438a9d20121d28.tar.gz
        URL_HASH SHA256=81b06ce7c19cf3b634b0271c23c361ba6ad8bf45fb8b036abbfeb4260ec1e126
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(raylib)
    find_package(Threads REQUIRED)
    add_executable(ataxx desktop.cpp)
    target_link_libraries(ataxx PRIVATE raylib Threads::Threads)
    target_compile_options(ataxx PRIVATE -Wall -Wextra -Wpedantic)
endif()
