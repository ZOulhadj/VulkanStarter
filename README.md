# Vulkan Starter
A basic Vulkan implementation that renders a simple triangle. If you see this, then
please be aware that the code is still a work in progress as I continue to clean up
the code and increase readability.

<img src=".gitassets/Application.png" width="500"/>

## Requirements
* [GLFW](https://glfw.org) - Cross-platform windowing
* [Vulkan SDK](https://vulkan.lunarg.com/) - Graphics API
* [CMake](https://cmake.org) - Project configuration/building

### Dependencies

As mentioned above, GLFW is used for the cross-platform windowing. This example project includes the GLFW repository as a git submodule.
When cloning the repository the full command is:

``` git clone --recurse-submodules git@github.com:ZOulhadj/VulkanStarter.git ```.

This will ensure that you clone both the actual repository and all dependencies which this case is only GLFW.

## Building

Before building, your system needs to have the VulkanSDK installed. Once installed and the SDK, ensure that the paths
with the ``` CompileShader.sh ``` file point to the correct SDK directory. This will ensure that the shaders can be compiled.

After the SDK and paths to the SDK have been set up, you can compile the project. Although CMake is used to create makefiles and
project file, you have to ensure that you have the correct programs installed. For example, if generating Visual Studio solution
files on Windows then this does not apply. However, if creating a makefile then g++ is required and thus, needs to be installed
and set in the system path.

Finally, create a folder called ``` Build ``` within the root directory. Then from within the ``` Build ``` folder, call ``` cmake ../ ```.
This will create an "out of source" build which will ensure that build files do not get mixed with the program source code.

### Shader Compiling
Unlike in OpenGL, shaders cannot be loaded as text directly to Vulkan. Instead,
Vulkan requires shaders to be in the SPIR-V (Standard, Portable Intermediate Representation-V)
intermediate language format when loaded.

This example project already includes precompiled SPIR-V shader files, however, if
you make any changes to the source text shader files then make sure to compile them.
There are a few ways to achieve this however, for simplicity, the Vulkan SDK provides a program
called ```glslc``` which converts the shader files at ```Source/Shaders``` from text to SPIR-V
and outputs them to the ```Build``` directory which then get loaded at runtime to Vulkan.
