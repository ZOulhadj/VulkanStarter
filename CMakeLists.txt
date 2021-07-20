cmake_minimum_required(VERSION 3.20.0)
project(VulkanStarter)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)

add_subdirectory(Dependencies/GLFW)

add_executable(VulkanStarter Source/Main.cpp)

target_link_libraries(VulkanStarter Vulkan::Vulkan glfw)
