cmake_minimum_required(VERSION 2.6)
project(vulkanplay)

add_executable(vulkanplay main.cpp)

install(TARGETS vulkanplay RUNTIME DESTINATION bin)
