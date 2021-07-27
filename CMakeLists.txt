cmake_minimum_required(VERSION 3.19.0)
project(VulkanStarter)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Build")

find_package(Vulkan REQUIRED)

add_subdirectory(Dependencies/GLFW)

add_executable(VulkanStarter Source/Main.cpp)

target_link_libraries(VulkanStarter Vulkan::Vulkan glfw)
