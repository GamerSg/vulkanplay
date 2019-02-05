cmake_minimum_required(VERSION 3.12)
cmake_policy(VERSION 3.12)
project(vulkanplay)
find_package(Vulkan REQUIRED)
find_package(Boost REQUIRED)
add_executable(vulkanplay main.cpp)
#GLFW Stuff
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("./dep/glfw")
#GLM
include_directories("./dep/glm")
#Boost
include_directories(${Boost_INCLUDE_DIRS}) 
#Link
target_link_libraries(vulkanplay Vulkan::Vulkan)
target_link_libraries(vulkanplay glfw)

install(TARGETS vulkanplay RUNTIME DESTINATION bin)
