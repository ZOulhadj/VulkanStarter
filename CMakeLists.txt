cmake_minimum_required(VERSION 3.20.2)
project(VulkanStarter)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(Dependencies/GLFW)

find_package(Vulkan REQUIRED)

add_executable(VulkanStarter Source/Main.cpp)

# todo: Find a way to link with shaderc for compile-time shader compiling
target_link_libraries(VulkanStarter glfw Vulkan::Vulkan)
