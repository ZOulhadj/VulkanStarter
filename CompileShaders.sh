glslCompilerLocation="../../VulkanSDK/1.2.182.0/macOS/bin/glslc"

vertexShaderSourceLocation="Source/Shaders/Triangle.vert"
fragmentShaderSourceLocation="Source/Shaders/Triangle.frag"

shaderOutputFolder="Build/"

$glslCompilerLocation $vertexShaderSourceLocation -o Build/Vert.spv
$glslCompilerLocation $fragmentShaderSourceLocation -o Build/Frag.spv