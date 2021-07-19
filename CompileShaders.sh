
# location to the SPIR-V compiler within the Vulkan SDK
glslCompilerLocation="../../VulkanSDK/1.2.182.0/macOS/bin/glslc"

# the text shader source file locations
vertexShaderSourceLocation="Source/Shaders/Triangle.vert"
fragmentShaderSourceLocation="Source/Shaders/Triangle.frag"

# compile the vertex and fragment shaders
$glslCompilerLocation $vertexShaderSourceLocation -o Build/Vert.spv
$glslCompilerLocation $fragmentShaderSourceLocation -o Build/Frag.spv