cmake_minimum_required(VERSION 3.1.3)

project(hello_opencv)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
# add_executable(show_img show_img.cpp)
