# location to the SPIR-V compiler within the Vulkan SDK
glslCompilerLocation="$HOME/Projects/Vulkan/1.2.182.0/x86_64/bin/glslc"

# the text shader source file locations
vertexShaderSourceLocation="Source/Shaders/Triangle.vert"
fragmentShaderSourceLocation="Source/Shaders/Triangle.frag"

# create shaders folder within the Build directory
mkdir -p Build/Shaders

# compile the vertex and fragment shaders
$glslCompilerLocation $vertexShaderSourceLocation -o Build/Shaders/Vert.spv
$glslCompilerLocation $fragmentShaderSourceLocation -o Build/Shaders/Frag.spv
