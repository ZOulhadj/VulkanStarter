workspace "VulkanStarter"
    location "ProjectFiles"
    architecture "x86_64"
    systemversion "latest"

    configurations
    {
        "Debug",
        "Release"
    }

    filter { "configurations:Debug" }
        symbols "On"
        defines { "_DEBUG" }
        debugdir "."

    filter { "configurations:Release" }
        optimize "On"
        defines { "_NDEBUG" }

    filter { }

project "VulkanStarter"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    

    targetdir ("Build/%{cfg.buildcfg}")
    objdir ("Build/%{cfg.buildcfg}/Intermediate")

    files 
    {
        "Source/**.hpp",
        "Source/**.cpp",
    }

    sysincludedirs
    {
        "Source",
        "Dependencies/**/Include",
        "../../VulkanSDK/1.2.182.0/macOS/include"
    }

    libdirs
    {
        "Dependencies/**/Libs",
        "../../VulkanSDK/1.2.182.0/macOS/lib"
    }

    filter { "system:windows" }
        links
        {
            "glfw3",
            "vulkan.1",
            "vulkan.1.2.182"
        }
    filter { "system:macosx" }
        links
        {
            "Cocoa.framework",
            "IOKit.framework",
            "glfw3",
            "vulkan.1",
            "vulkan.1.2.182"
        }

    filter {}