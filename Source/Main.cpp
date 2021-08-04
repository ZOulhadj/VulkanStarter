// todo: implement VK_LAYER_LUNARG_monitor valiation layer to display FPS
// todo: add debug message callback specifically for instance creation and deletion
/*
 * todo: swapchain images are not UniqueImage because the get images function only returns Image.
 * todo: need to find out more.
 */

#include <iostream>
#include <vector>
#include <optional>
#include <set>
#include <fstream>
#include <array>
#include <utility>

#define VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>

//#include <shaderc/shaderc.hpp>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    [[nodiscard]] bool IsComplete() const
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails
{
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

class Window
{
private:
    GLFWwindow* m_Window;
    std::string m_Title;
    int m_Width, m_Height;
public:
    Window(std::string title, int width, int height)
        : m_Window(nullptr), m_Width(width), m_Height(height), m_Title(std::move(title))
    {
        assert(m_Width > 0 && m_Height > 0);
    }

    ~Window()
    {
        glfwDestroyWindow(m_Window);
    }

    [[nodiscard]] bool CreateWindow()
    {
        // we will not be using OpenGL
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, false);

        m_Window = glfwCreateWindow(m_Width, m_Height, m_Title.c_str(), nullptr, nullptr);

        return m_Window;
    }

    [[nodiscard]] bool ShouldClose() const { return glfwWindowShouldClose(m_Window); }

    [[nodiscard]] GLFWwindow* Get() const { return m_Window; }
    [[nodiscard]] int GetWidth() const
    {
        int width, height;
        glfwGetFramebufferSize(m_Window, &width, &height);

        return width;
    }

    [[nodiscard]] int GetHeight() const
    {
        int width, height;
        glfwGetFramebufferSize(m_Window, &width, &height);

        return height;
    }

};

static VKAPI_ATTR vk::Bool32 VKAPI_CALL DebugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        vk::DebugUtilsMessageTypeFlagsEXT messageType,
        const vk::DebugUtilsMessengerCallbackDataEXT* callback,
        void* userData);

bool CheckValidationLayerSupport(const std::vector<const char*>& validationLayers);
std::optional<std::vector<char>> LoadShader(const std::string& path);
bool CheckDeviceExtensionSupport(const vk::PhysicalDevice& physicalDevice, const std::vector<const char*>& deviceExtensions);
QueueFamilyIndices CheckQueueFamilies(const vk::PhysicalDevice& physicalDevice, const vk::UniqueSurfaceKHR& surface);
SwapChainSupportDetails CheckSwapChainSupport(const vk::PhysicalDevice& physicalDevice, const vk::UniqueSurfaceKHR& surface);
vk::SurfaceFormatKHR SelectSwapChainSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
vk::PresentModeKHR SelectSwapChainPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
vk::Extent2D SelectSwapChainExtent(int width, int height, const vk::SurfaceCapabilitiesKHR& capabilities);
vk::UniqueShaderModule CreateShaderModule(const vk::UniqueDevice& device, const std::vector<char>& shaderCode);

int main()
{

    if (!glfwInit())
    {
        std::cout << "Failed to initialise GLFW\n";
        return 0;
    }

    Window window("Vulkan Application", 800, 600);
    if (!window.CreateWindow())
    {
        std::cout << "Failed to create GLFW window\n";
        return 0;
    }



    // note: VK_KHR_get_physical_device_properties2 is required for macOS due to MoltenVK
    std::vector<const char*> instanceExtensions =
    {
        #ifdef __APPLE__
        "VK_KHR_get_physical_device_properties2",
        #endif

        #ifndef NDEBUG
        "VK_EXT_debug_utils",
        #endif
    };


    // note: VK_KHR_portability_subset is required for macOS due to MoltenVK
    const std::vector<const char*> deviceExtensions =
    {
    #ifdef __APPLE__
        "VK_KHR_portability_subset",
    #endif
        "VK_KHR_swapchain",
    };

#ifdef NDEBUG
    const std::vector<const char*> validationLayers;
#else
    const std::vector<const char*> validationLayers =
    {
        "VK_LAYER_KHRONOS_validation",
    };
#endif


    vk::UniqueInstance g_Instance;

#ifndef NDEBUG
    vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> g_DebugMessenger;
#endif

    vk::UniqueSurfaceKHR g_Surface;

    vk::PhysicalDevice g_PhysicalDevice;
    vk::UniqueDevice g_Device;
    vk::Queue g_GraphicsQueue;
    vk::Queue g_PresentQueue;
    vk::UniqueSwapchainKHR g_SwapChain;

    // todo: Image is used for getswapchainImages, since there is no UniqueImage?
    std::vector<vk::Image> g_SwapChainImages;
    vk::Format g_SwapChainImageFormat;
    vk::Extent2D g_SwapChainExtent;
    std::vector<vk::UniqueImageView> g_SwapChainImageViews;

    vk::UniqueRenderPass g_RenderPass;
    vk::UniquePipelineLayout g_PipelineLayout;
    std::vector<vk::UniquePipeline> g_GraphicsPipeline;

    std::vector<vk::UniqueFramebuffer> g_SwapChainFramebuffers;

    vk::UniqueCommandPool g_CommandPool;
    std::vector<vk::UniqueCommandBuffer> g_CommandBuffers;


    const int MAX_FRAMES_IN_FLIGHT = 2;

    std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
    std::vector<vk::UniqueFence> inFlightFences;
    std::vector<vk::Fence*> imagesInFlight;

    std::size_t currentFrame = 0;


    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    const std::vector<const char*> glfwExtensionsList(glfwExtensions, glfwExtensions + glfwExtensionCount);
    instanceExtensions.insert(instanceExtensions.end(), glfwExtensionsList.begin(), glfwExtensionsList.end());

    // check validation/debugging layer support in debug mode
#ifndef NDEBUG
    if (!CheckValidationLayerSupport(validationLayers))
    {
        std::cout << "Validation layers requested but not available\n";
        return 0;
    }
#endif



    vk::ApplicationInfo applicationInfo
    {
        .pApplicationName = "Application Name",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0
    };

    vk::InstanceCreateInfo instanceInfo
    {
        .pApplicationInfo = &applicationInfo,
        .enabledLayerCount = static_cast<uint32_t>(validationLayers.size()),
        .ppEnabledLayerNames = validationLayers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size()),
        .ppEnabledExtensionNames = instanceExtensions.data()
    };

    g_Instance = vk::createInstanceUnique(instanceInfo);
    if (!g_Instance)
    {
        std::cout << "Failed to create Vulkan instance\n";
        return 0;
    }

#ifndef NDEBUG
    using SeverityFlagBit = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using TypeFlagBit = vk::DebugUtilsMessageTypeFlagBitsEXT;

    vk::DebugUtilsMessengerCreateInfoEXT debugInfo
    {
        .messageSeverity = SeverityFlagBit::eVerbose | SeverityFlagBit::eWarning | SeverityFlagBit::eError,
        // note: TypeFlagBit::eGeneral was used here before
        .messageType = TypeFlagBit::eValidation | TypeFlagBit::ePerformance,
        .pfnUserCallback = reinterpret_cast<PFN_vkDebugUtilsMessengerCallbackEXT>(DebugCallback),
        .pUserData = nullptr
    };

    vk::DispatchLoaderDynamic dynamicLoader(g_Instance.get(), vkGetInstanceProcAddr);
    g_DebugMessenger = g_Instance->createDebugUtilsMessengerEXTUnique(debugInfo, nullptr, dynamicLoader);
    if (!g_DebugMessenger)
    {
        std::cout << "Failed to create debug messenger callback\n";
        return 0;
    }
#endif


    // todo: check if there is a way to handle the surface reference
    VkSurfaceKHR glfwSurface = {};
    if (glfwCreateWindowSurface(g_Instance.get(), window.Get(), nullptr, &glfwSurface) != VK_SUCCESS)
    {
        std::cout << "Failed to create Vulkan window surface\n";
        return 0;
    }
    g_Surface = vk::UniqueSurfaceKHR(glfwSurface, g_Instance.get());

    // get the total number of GPU's that support Vulkan on the system
    std::vector<vk::PhysicalDevice> physicalDevices = g_Instance->enumeratePhysicalDevices();
    if (physicalDevices.empty())
    {
        std::cout << "No GPU's found with Vulkan support\n";
        return 0;
    }

    // check if GPU's support required Vulkan features
    QueueFamilyIndices indices = {};
    std::vector<std::size_t> discreteDevicePositions, integratedDevicePositions;
    for (std::size_t i = 0; i < physicalDevices.size(); ++i)
    {
        indices = CheckQueueFamilies(physicalDevices[i], g_Surface);

        bool extensionsSupported = CheckDeviceExtensionSupport(physicalDevices[i], deviceExtensions);
        if (!extensionsSupported)
            continue;

        SwapChainSupportDetails swapChainSupport = CheckSwapChainSupport(physicalDevices[i], g_Surface);
        bool swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();


        if (!(indices.IsComplete() && swapChainAdequate))
            continue;

        // find out what type of GPU it is and save its position index
        vk::PhysicalDeviceProperties physicalDeviceProperties = physicalDevices[i].getProperties();
        if (physicalDeviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
            discreteDevicePositions.push_back(i);
        else if (physicalDeviceProperties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
            integratedDevicePositions.push_back(i);
    }

    // If the system has a compatible discrete GPU then use that else use an intergrated GPU
    if (!discreteDevicePositions.empty())
        g_PhysicalDevice = vk::PhysicalDevice(physicalDevices[discreteDevicePositions[0]]);
    else if (!integratedDevicePositions.empty())
        g_PhysicalDevice = vk::PhysicalDevice(physicalDevices[integratedDevicePositions[0]]);

    if (!g_PhysicalDevice)
    {
        std::cout << "No GPU's found with required Vulkan features\n";
        return 0;
    }

    std::vector<vk::DeviceQueueCreateInfo> deviceQueueInfos;
    std::set<uint32_t> uniqueQueueFamilies =
    {
        indices.graphicsFamily.value(),
        indices.presentFamily.value()
    };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies)
    {
        vk::DeviceQueueCreateInfo deviceQueueInfo =
        {
            .queueFamilyIndex = queueFamily,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority
        };

        deviceQueueInfos.push_back(deviceQueueInfo);
    }

    vk::PhysicalDeviceFeatures physicalDeviceFeatures;

    vk::DeviceCreateInfo deviceInfo =
    {
        .queueCreateInfoCount = static_cast<uint32_t>(deviceQueueInfos.size()),
        .pQueueCreateInfos = deviceQueueInfos.data(),
        .enabledLayerCount = static_cast<uint32_t>(validationLayers.size()),
        .ppEnabledLayerNames = validationLayers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
        .pEnabledFeatures = &physicalDeviceFeatures
    };

    g_Device = g_PhysicalDevice.createDeviceUnique(deviceInfo);
    if (!g_Device)
    {
        std::cout << "Failed to create Vulkan device\n";
        return 0;
    }

    g_Device->getQueue(indices.graphicsFamily.value(), 0, &g_GraphicsQueue);
    g_Device->getQueue(indices.presentFamily.value(), 0, &g_PresentQueue);


    // create swap chain
    SwapChainSupportDetails swapChainSupport = CheckSwapChainSupport(g_PhysicalDevice, g_Surface);

    vk::SurfaceFormatKHR surfaceFormat = SelectSwapChainSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode = SelectSwapChainPresentMode(swapChainSupport.presentModes);
    vk::Extent2D extent = SelectSwapChainExtent(window.GetWidth(), window.GetHeight(), swapChainSupport.capabilities);


    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        imageCount = swapChainSupport.capabilities.maxImageCount;


    std::array<uint32_t, 2> queueFamilyIndices =
    {
        indices.graphicsFamily.value(),
        indices.presentFamily.value()
    };


    vk::SharingMode imageSharingMode;
    uint32_t queueFamilyIndexCount;
    uint32_t* pQueueFamilyIndices;

    if (indices.graphicsFamily != indices.presentFamily)
    {
        imageSharingMode = vk::SharingMode::eConcurrent;
        queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size());
        pQueueFamilyIndices = queueFamilyIndices.data();
    }
    else
    {
        imageSharingMode = vk::SharingMode::eExclusive;
        queueFamilyIndexCount = 0; // Optional
        pQueueFamilyIndices = nullptr; // Optional
    }

    vk::SwapchainCreateInfoKHR swapchainInfo
    {
        .surface = g_Surface.get(),
        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = imageSharingMode,
        .queueFamilyIndexCount = queueFamilyIndexCount,
        .pQueueFamilyIndices = pQueueFamilyIndices,
        .preTransform = swapChainSupport.capabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = presentMode,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE
    };


    g_SwapChain = g_Device->createSwapchainKHRUnique(swapchainInfo);
    if (!g_SwapChain)
    {
        std::cout << "Failed to create Vulkan swap chain\n";
        return 0;
    }

    g_SwapChainImages = g_Device->getSwapchainImagesKHR(g_SwapChain.get());
    g_SwapChainImageFormat = surfaceFormat.format;
    g_SwapChainExtent = extent;

    g_SwapChainImageViews.resize(g_SwapChainImages.size());

    for (std::size_t i = 0; i < g_SwapChainImageViews.size(); ++i)
    {
        vk::ImageViewCreateInfo imageViewInfo
        {
            .image = g_SwapChainImages[i],
            .viewType = vk::ImageViewType::e2D,
            .format = g_SwapChainImageFormat,
            .components = { vk::ComponentSwizzle::eIdentity },
            .subresourceRange =
            {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        g_SwapChainImageViews[i] = g_Device->createImageViewUnique(imageViewInfo);
        if (!g_SwapChainImageViews[i])
        {
            std::cout << "Failed to create Vulkan image view\n";
            return 0;
        }
    }


    // render pass
    vk::AttachmentDescription attachmentDescription
    {
        .format = g_SwapChainImageFormat,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = vk::ImageLayout::ePresentSrcKHR
    };

    vk::AttachmentReference colorAttachmentReference
    {
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal
    };

    vk::SubpassDescription subpassDescription
    {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentReference
    };

    vk::SubpassDependency subpassDependency
    {

        // todo: find out if subpass external has a proper enum type
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite
    };

    vk::RenderPassCreateInfo renderPassInfo
    {
        .attachmentCount = 1,
        .pAttachments = &attachmentDescription,
        .subpassCount = 1,
        .pSubpasses = &subpassDescription,
        .dependencyCount = 1,
        .pDependencies = &subpassDependency
    };

    g_RenderPass = g_Device->createRenderPassUnique(renderPassInfo);
    if (!g_RenderPass)
    {
        std::cout << "Failed to create render pass\n";
        return 0;
    }



    // initialisation of graphics pipeline
    std::optional<std::vector<char>> vertexShaderCode = LoadShader("../Shaders/Vert.spv");
    std::optional<std::vector<char>> fragmentShaderCode = LoadShader("../Shaders/Frag.spv");

    if (!vertexShaderCode || !fragmentShaderCode)
    {
        std::cout << "Failed to load vertex or fragment shader\n";
        return 0;
    }

    vk::UniqueShaderModule vertexShaderModule = CreateShaderModule(g_Device, vertexShaderCode.value());
    vk::UniqueShaderModule fragmentShaderModule = CreateShaderModule(g_Device, fragmentShaderCode.value());

    vk::PipelineShaderStageCreateInfo vertexStageInfo =
    {
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = vertexShaderModule.get(),
        .pName = "main"
    };

    vk::PipelineShaderStageCreateInfo fragmentStageInfo =
    {
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = fragmentShaderModule.get(),
        .pName = "main"
    };

    std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages =
    {
        vertexStageInfo,
        fragmentStageInfo
    };


    vk::PipelineVertexInputStateCreateInfo vertexInputInfo =
    {
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = nullptr
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly =
    {
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = VK_FALSE
    };

    vk::Viewport viewport =
    {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(g_SwapChainExtent.width),
        .height = static_cast<float>(g_SwapChainExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };

    vk::Rect2D scissor
    {
        .offset = { 0, 0 },
        .extent = g_SwapChainExtent
    };

    vk::PipelineViewportStateCreateInfo viewportState =
    {
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor
    };

    vk::PipelineRasterizationStateCreateInfo rasterizer =
    {
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eClockwise,
        .depthBiasEnable = false,
        .lineWidth = 1.0f,
    };


    vk::PipelineMultisampleStateCreateInfo multisampling =
    {
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = false,
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment =
    {
        .blendEnable = false,
        .colorWriteMask =
        {
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA
        }
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending =
    {
        .logicOpEnable = false,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
        .blendConstants = std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f }
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo =
    {
        .setLayoutCount = 0,
        .pushConstantRangeCount = 0,
    };

    g_PipelineLayout = g_Device->createPipelineLayoutUnique(pipelineLayoutInfo);
    if (!g_PipelineLayout)
    {
        std::cout << "Failed to create pipeline layout\n";
        return 0;
    }

    vk::GraphicsPipelineCreateInfo graphicsPipelineCreateInfo =
    {
        .stageCount = static_cast<uint32_t>(shaderStages.size()),
        .pStages = shaderStages.data(),
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = nullptr,
        .pColorBlendState = &colorBlending,
        .pDynamicState = nullptr,
        .layout = g_PipelineLayout.get(),
        .renderPass = g_RenderPass.get(),
        .subpass = 0,
        .basePipelineHandle = nullptr,

    };

    g_GraphicsPipeline = g_Device->createGraphicsPipelinesUnique(nullptr, graphicsPipelineCreateInfo, nullptr).value;

    if (!g_GraphicsPipeline[0])
    {
        std::cout << "Failed to create graphics pipeline\n";
        return 0;
    }


/*    vertexShaderModule.release();
    fragmentShaderModule.release();*/
   /* g_Device->destroyShaderModule(vertexShaderModule.get(), nullptr);
    g_Device->destroyShaderModule(fragmentShaderModule.get(), nullptr);*/

    g_SwapChainFramebuffers.resize(g_SwapChainImageViews.size());

    for (std::size_t i = 0; i < g_SwapChainImageViews.size(); ++i)
    {
        std::array<vk::ImageView, 1> attachments = { g_SwapChainImageViews[i].get() };

        vk::FramebufferCreateInfo framebufferCreateInfo =
        {
            .renderPass = g_RenderPass.get(),
            .attachmentCount = attachments.size(),
            .pAttachments = attachments.data(),
            .width = g_SwapChainExtent.width,
            .height = g_SwapChainExtent.height,
            .layers = 1
        };

        g_SwapChainFramebuffers[i] = g_Device->createFramebufferUnique(framebufferCreateInfo, nullptr);
        if (!g_SwapChainFramebuffers[i])
        {
            std::cout << "Failed to create Vulkan framebuffer\n";
            return 0;
        }
    }


    vk::CommandPoolCreateInfo commandPoolCreateInfo =
    {
        .queueFamilyIndex = indices.graphicsFamily.value()
    };

    g_CommandPool = g_Device->createCommandPoolUnique(commandPoolCreateInfo, nullptr);
    if (!g_CommandPool)
    {
        std::cout << "Failed to created Vulkan command pool\n";
        return 0;
    }

    g_CommandBuffers.resize(g_SwapChainFramebuffers.size());
    vk::CommandBufferAllocateInfo cbAllocateInfo =
    {
        .commandPool = g_CommandPool.get(),
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = static_cast<uint32_t>(g_CommandBuffers.size())
    };

    g_CommandBuffers = g_Device->allocateCommandBuffersUnique(cbAllocateInfo);

    // todo: find out if this is needed
    if (g_CommandBuffers.empty())
    {
        std::cout << "Command buffers are empty\n";
        return 0;
    }


    for (std::size_t i = 0; i < g_CommandBuffers.size(); ++i)
    {
        vk::CommandBufferBeginInfo cbBeginInfo = {};

        if (g_CommandBuffers[i]->begin(&cbBeginInfo) != vk::Result::eSuccess)
        {
            std::cout << "Failed to begin recording Vulkan command buffers\n";
            return 0;
        }


        vk::ClearValue clearColor(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f });

        vk::RenderPassBeginInfo renderPassBeginInfo =
        {
            .renderPass = g_RenderPass.get(),
            .framebuffer = g_SwapChainFramebuffers[i].get(),
            .renderArea = {{ 0, 0 }, g_SwapChainExtent },
            .clearValueCount = 1,
            .pClearValues = &clearColor
        };

        g_CommandBuffers[i]->beginRenderPass(&renderPassBeginInfo, vk::SubpassContents::eInline);
        g_CommandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics, g_GraphicsPipeline[0].get());

        g_CommandBuffers[i]->draw(3, 1, 0, 0);

        g_CommandBuffers[i]->endRenderPass();

        g_CommandBuffers[i]->end();

    }

    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(g_SwapChainImages.size());

    vk::SemaphoreCreateInfo semaphoreCreateInfo = {};
    vk::FenceCreateInfo fenceCreateInfo = { .flags = vk::FenceCreateFlagBits::eSignaled };

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        imageAvailableSemaphores[i] = g_Device->createSemaphoreUnique(semaphoreCreateInfo, nullptr);
        if (!imageAvailableSemaphores[i])
        {
            std::cout << "Failed to create Vulkan image semaphore\n";
            return 0;
        }

        renderFinishedSemaphores[i] = g_Device->createSemaphoreUnique(semaphoreCreateInfo, nullptr);
        if (!renderFinishedSemaphores[i])
        {
            std::cout << "Failed to create Vulkan render semaphore\n";
            return 0;
        }

        inFlightFences[i] = g_Device->createFenceUnique(fenceCreateInfo, nullptr);
        if (!inFlightFences[i])
        {
            std::cout << "Failed to create Vulkan in flight fence\n";
            return 0;
        }

    }

    while (!window.ShouldClose())
    {
        if (g_Device->waitForFences(1, &inFlightFences[currentFrame].get(), VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
        {
            std::cout << "Unable to wait for fence?\n";
            return 0;
        }

        uint32_t imageIndex;
        vk::Result acquireNextImageResult = g_Device->acquireNextImageKHR(
                g_SwapChain.get(),
                UINT64_MAX,
                imageAvailableSemaphores[currentFrame].get(),
                nullptr,
                &imageIndex);
        if (acquireNextImageResult != vk::Result::eSuccess)
        {
            std::cout << "Failed to acquire next image\n";
            return 0;
        }

        if (imagesInFlight[imageIndex])
        {
            if (g_Device->waitForFences(1, imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
            {
                std::cout << "Error for 'waitForFences'.\n";
                return 0;
            }

        }


        imagesInFlight[imageIndex] = &inFlightFences[currentFrame].get();



        vk::SubmitInfo submitInfo = {};


        std::array<vk::Semaphore, 1> waitSemaphores =
        {
            imageAvailableSemaphores[currentFrame].get()
        };

        std::array<vk::PipelineStageFlags, 1> waitStages =
        {
            vk::PipelineStageFlagBits::eColorAttachmentOutput
        };

        submitInfo.waitSemaphoreCount = waitSemaphores.size();
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &g_CommandBuffers[imageIndex].get();

        std::array<vk::Semaphore, 1> signalSemaphores =
        {
            renderFinishedSemaphores[currentFrame].get()
        };
        submitInfo.signalSemaphoreCount = signalSemaphores.size();
        submitInfo.pSignalSemaphores = signalSemaphores.data();

        if (g_Device->resetFences(1, &inFlightFences[currentFrame].get()) != vk::Result::eSuccess)
        {
            std::cout << "Unable to reset fence\n";
            return 0;
        }

        if (g_GraphicsQueue.submit(1, &submitInfo, inFlightFences[currentFrame].get()) != vk::Result::eSuccess)
        {
            std::cout << "Failed to submit Vulkan draw command buffer\n";
            return 0;
        }

        vk::PresentInfoKHR presentInfo =
        {
            .waitSemaphoreCount = signalSemaphores.size(),
            .pWaitSemaphores = signalSemaphores.data()
        };

        std::array<vk::SwapchainKHR, 1> swapChains = { g_SwapChain.get() };
        presentInfo.swapchainCount = swapChains.size();
        presentInfo.pSwapchains = swapChains.data();
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        if (g_PresentQueue.presentKHR(&presentInfo) != vk::Result::eSuccess)
        {
            std::cout << "Unable to present\n";
            return 0;
        }


        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;


        glfwPollEvents();
    }

    vkDeviceWaitIdle(g_Device.get());

    // todo: terminating with uncaught exception of type std::__1::system_error: mutex lock failed: Invalid argument
    // todo: due to vk::UniqueFence



    glfwTerminate();

    return 0;
}


static VKAPI_ATTR vk::Bool32 VKAPI_CALL DebugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        vk::DebugUtilsMessageTypeFlagsEXT messageType,
        const vk::DebugUtilsMessengerCallbackDataEXT* callback,
        void* userData)
{

    std::cerr << callback->pMessage << "\n";

    return VK_FALSE;
}

bool CheckValidationLayerSupport(const std::vector<const char*>& validationLayers)
{
    std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

    for (const std::string& layerName : validationLayers)
    {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers)
        {
            if (layerName == layerProperties.layerName)
            {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
            return false;
    }

    return true;
}


std::optional<std::vector<char>> LoadShader(const std::string& path)
{
    std::ifstream shader(path, std::ios::ate | std::ios::binary);

    if (!shader.is_open())
        return {};

    size_t fileSize = (size_t) shader.tellg();
    std::vector<char> buffer(fileSize);
    shader.seekg(0);
    shader.read(buffer.data(), fileSize);

    return buffer;
}


bool CheckDeviceExtensionSupport(const vk::PhysicalDevice& physicalDevice, const std::vector<const char*>& deviceExtensions)
{
    std::vector<vk::ExtensionProperties> availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const vk::ExtensionProperties& extension : availableExtensions)
    requiredExtensions.erase(extension.extensionName);

    return requiredExtensions.empty();
}


QueueFamilyIndices CheckQueueFamilies(const vk::PhysicalDevice& physicalDevice, const vk::UniqueSurfaceKHR& surface)
{
    QueueFamilyIndices indices = {};
    std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();

    int i = 0;
    vk::Bool32 presentSupported = false;
    for (const auto& queueFamily : queueFamilies)
    {

        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
            indices.graphicsFamily = i;

        if (physicalDevice.getSurfaceSupportKHR(i, surface.get(), &presentSupported) != vk::Result::eSuccess)
            continue;

        if (presentSupported)
            indices.presentFamily = i;

        if (indices.IsComplete())
            break;

        i++;
    }

    return indices;
}


SwapChainSupportDetails CheckSwapChainSupport(const vk::PhysicalDevice& physicalDevice, const vk::UniqueSurfaceKHR& surface)
{
    SwapChainSupportDetails details = {};
    details.capabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface.get());
    details.formats = physicalDevice.getSurfaceFormatsKHR(surface.get());
    details.presentModes = physicalDevice.getSurfacePresentModesKHR(surface.get());

    return details;
}

vk::SurfaceFormatKHR SelectSwapChainSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    for (const auto& format : availableFormats)
    {
        if (format.format == vk::Format::eB8G8R8A8Srgb &&
        format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)

            return format;
    }

    return availableFormats[0];
}

vk::PresentModeKHR SelectSwapChainPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
{
    for (const auto& format : availablePresentModes)
    {
        if (format == vk::PresentModeKHR::eMailbox)
            return format;
    }

    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D SelectSwapChainExtent(int width, int height, const vk::SurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != UINT32_MAX)
        return capabilities.currentExtent;

    vk::Extent2D actualExtent =
    {
        .width = static_cast<uint32_t>(width),
        .height = static_cast<uint32_t>(height)
    };

    actualExtent.width = std::clamp(actualExtent.width,
                                    capabilities.minImageExtent.width,
                                    capabilities.maxImageExtent.width);

    actualExtent.height = std::clamp(actualExtent.height,
                                     capabilities.minImageExtent.height,
                                     capabilities.maxImageExtent.height);

    return actualExtent;
}

vk::UniqueShaderModule CreateShaderModule(const vk::UniqueDevice& device, const std::vector<char>& shaderCode)
{
    vk::ShaderModuleCreateInfo shaderInfo
    {
        .codeSize = shaderCode.size(),
        .pCode = reinterpret_cast<const uint32_t*>(shaderCode.data())
    };

    vk::UniqueShaderModule shaderModule = device->createShaderModuleUnique(shaderInfo, nullptr);

    if (!shaderModule)
    {
        std::cout << "Failed to create shader module\n";
        return {};
    }

    return shaderModule;
}
