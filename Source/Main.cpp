// todo: implement VK_LAYER_LUNARG_monitor valiation layer to display FPS
// todo: add message callback for debugging


#include <iostream>
#include <vector>
#include <optional>
#include <set>
#include <fstream>
#include <array>

#define VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>


#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>


GLFWwindow* window = nullptr;
const uint16_t WIDTH = 800;
const uint16_t HEIGHT = 600;

VkResult result = VK_SUCCESS;

#ifdef NDEBUG
    const std::vector<const char*> deviceExtensions;
    const std::vector<const char*> validationLayers;
#else
    const std::vector<const char*> deviceExtensions =
    {
        "VK_KHR_swapchain",
    };

    const std::vector<const char*> validationLayers =
    {
        "VK_LAYER_KHRONOS_validation",
    };
#endif



vk::Instance instance;
vk::SurfaceKHR surface;

vk::PhysicalDevice physicalDevice;
vk::Device device;
vk::Queue graphicsQueue;
vk::Queue presentQueue;
vk::SwapchainKHR swapChain;

std::vector<vk::Image> swapChainImages;
vk::Format swapChainImageFormat;
vk::Extent2D swapChainExtent;
std::vector<vk::ImageView> swapChainImageViews;

vk::RenderPass renderPass;
vk::PipelineLayout pipelineLayout;
vk::Pipeline graphicsPipeline;

std::vector<vk::Framebuffer> swapChainFramebuffers;

vk::CommandPool commandPool;
std::vector<vk::CommandBuffer> commandBuffers;


const int MAX_FRAMES_IN_FLIGHT = 2;

std::vector<vk::Semaphore> imageAvailableSemaphores;
std::vector<vk::Semaphore> renderFinishedSemaphores;
std::vector<vk::Fence> inFlightFences;
std::vector<vk::Fence> imagesInFlight;

std::size_t currentFrame = 0;


void InputCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window, true);
}

static std::vector<char> LoadShader(const std::string& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        std::cout << "Failed to load " << path << " shader file\n";
        return {};
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);

    return buffer;
}

struct SwapChainSupportDetails
{
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

bool CheckDeviceExtensionSupport(const vk::PhysicalDevice& device)
{
    // todo: change the way we do this...
    std::vector<vk::ExtensionProperties> availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const vk::ExtensionProperties& extension : availableExtensions)
        requiredExtensions.erase(extension.extensionName);

    return requiredExtensions.empty();
}


bool CheckValidationLayerSupport()
{
    std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

    for (const std::string& layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (layerName == layerProperties.layerName) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
            return false;
    }

    return true;
}

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    [[nodiscard]] bool IsComplete() const
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

QueueFamilyIndices CheckQueueFamilies(const vk::PhysicalDevice& device)
{
    QueueFamilyIndices indices;
    std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

    int i = 0;
    vk::Bool32 presentSupported = false;
    for (const auto& queueFamily : queueFamilies)
    {

        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
            indices.graphicsFamily = i;

        if (device.getSurfaceSupportKHR(i, surface, &presentSupported) != vk::Result::eSuccess)
            continue;

        if (presentSupported)
            indices.presentFamily = i;

        if (indices.IsComplete())
            break;

        i++;
    }

    return indices;
}


SwapChainSupportDetails CheckSwapChainSupport(const vk::PhysicalDevice& device)
{
    SwapChainSupportDetails details;

    uint32_t formatCount;
    if (device.getSurfaceFormatsKHR(surface, &formatCount, nullptr) != vk::Result::eSuccess)
    {
        std::cout << "Unable to obtain surface format count\n";
        return {};
    }


    if (formatCount != 0)
    {
        details.formats.resize(formatCount);
        if (device.getSurfaceFormatsKHR(surface, &formatCount, details.formats.data()) != vk::Result::eSuccess)
        {
            std::cout << "Unable to obtain surface format count\n";
            return {};
        }
    }

    uint32_t presentModeCount;

    if (device.getSurfacePresentModesKHR(surface, &presentModeCount, nullptr) != vk::Result::eSuccess)
    {
        std::cout << "Unable to obtain surface present mode count\n";
        return {};
    }

    if (presentModeCount != 0)
    {
        details.presentModes.resize(presentModeCount);
        if (device.getSurfacePresentModesKHR(surface, &presentModeCount, details.presentModes.data()) != vk::Result::eSuccess)
        {
            std::cout << "Unable to obtain surface present mode count\n";
            return {};
        }
    }

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

vk::Extent2D SelectSwapChainExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != UINT32_MAX)
        return capabilities.currentExtent;

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

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

bool CheckDeviceSuitability(const vk::PhysicalDevice& device)
{
    QueueFamilyIndices indices = CheckQueueFamilies(device);

    bool extensionsSupported = CheckDeviceExtensionSupport(device);
    if (!extensionsSupported)
        return false;

    SwapChainSupportDetails swapChainSupport = CheckSwapChainSupport(device);
    bool swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();


    return indices.IsComplete() && swapChainAdequate;
}


vk::ShaderModule CreateShaderModule(const std::vector<char>& shaderCode)
{
    vk::ShaderModuleCreateInfo shaderInfo
    {
        .codeSize = shaderCode.size(),
        .pCode = reinterpret_cast<const uint32_t*>(shaderCode.data())
    };

    vk::ShaderModule shaderModule = {};
    if (device.createShaderModule(&shaderInfo, nullptr, &shaderModule) != vk::Result::eSuccess)
    {
        std::cout << "Failed to create shader module\n";
        return VK_NULL_HANDLE;
    }

    return shaderModule;
}

int main()
{
    if (!glfwInit())
    {
        std::cout << "Failed to initialise GLFW\n";
        return 0;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Starter", nullptr, nullptr);
    if (!window)
    {
        std::cout << "Failed to create GLFW window\n";
        return 0;
    }

    glfwSetKeyCallback(window, InputCallback);

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    // initialise Vulkan
#ifndef NDEBUG
    if (!CheckValidationLayerSupport())
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
        .enabledExtensionCount = glfwExtensionCount,
        .ppEnabledExtensionNames = glfwExtensions
    };

    instance = vk::createInstance(instanceInfo);
    if (!instance)
    {
        std::cout << "Failed to create Vulkan instance\n";
        return 0;
    }

    // todo: check if there is a way to handle the surface reference
    VkSurfaceKHR glfwSurface = {};
    if (glfwCreateWindowSurface(VkInstance(instance), window, nullptr, &glfwSurface) != VK_SUCCESS)
    {
        std::cout << "Failed to create Vulkan window surface\n";
        return 0;
    }
    surface = vk::SurfaceKHR(glfwSurface);

    // get the total number of GPU's that support Vulkan on the system
    std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
    if (physicalDevices.empty())
    {
        std::cout << "No GPU's found with Vulkan support\n";
        return 0;
    }

    // check if GPU's support required Vulkan features
    for (const auto& physicalDev : physicalDevices)
    {
        if (!CheckDeviceSuitability(physicalDev))
            continue;

        // todo: at the moment the first GPU will be selected. But instead I should
        // first check for a dedicated GPU and if none is found then fallback to an
        // integrated GPU.
        physicalDevice = physicalDev;
        break;
    }

    if (!physicalDevice)
    {
        std::cout << "No GPU's found with required Vulkan features\n";
        return 0;
    }


    QueueFamilyIndices indices = CheckQueueFamilies(physicalDevice);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
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

        queueCreateInfos.push_back(deviceQueueInfo);
    }

    vk::PhysicalDeviceFeatures physicalDeviceFeatures;

    vk::DeviceCreateInfo deviceInfo =
    {
        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledLayerCount = static_cast<uint32_t>(validationLayers.size()),
        .ppEnabledLayerNames = validationLayers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
        .pEnabledFeatures = &physicalDeviceFeatures
    };

    device = physicalDevice.createDevice(deviceInfo);
    if (!device)
    {
        std::cout << "Failed to create Vulkan device\n";
        return 0;
    }

    device.getQueue(indices.graphicsFamily.value(), 0, &graphicsQueue);
    device.getQueue(indices.presentFamily.value(), 0, &presentQueue);


    // create swap chain
    SwapChainSupportDetails swapChainSupport = CheckSwapChainSupport(physicalDevice);

    vk::SurfaceFormatKHR surfaceFormat = SelectSwapChainSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode = SelectSwapChainPresentMode(swapChainSupport.presentModes);
    vk::Extent2D extent = SelectSwapChainExtent(swapChainSupport.capabilities);


    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        imageCount = swapChainSupport.capabilities.maxImageCount;



    QueueFamilyIndices swapChainIndices = CheckQueueFamilies(physicalDevice);
    std::array<uint32_t, 2> queueFamilyIndices =
    {
        swapChainIndices.graphicsFamily.value(),
        swapChainIndices.presentFamily.value()
    };


    vk::SharingMode imageSharingMode;
    uint32_t queueFamilyIndexCount;
    uint32_t* pQueueFamilyIndices;

    if (swapChainIndices.graphicsFamily != swapChainIndices.presentFamily)
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
        .surface = surface,
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


    if (device.createSwapchainKHR(&swapchainInfo, nullptr, &swapChain) != vk::Result::eSuccess)
    {
        std::cout << "Failed to create Vulkan swap chain\n";
        return 0;
    }

    device.getSwapchainImagesKHR(swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    device.getSwapchainImagesKHR(swapChain, &imageCount, swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;

    swapChainImageViews.resize(swapChainImages.size());

    for (std::size_t i = 0; i < swapChainImageViews.size(); ++i)
    {
        vk::ImageViewCreateInfo imageViewInfo
        {
            .image = swapChainImages[i],
            .viewType = vk::ImageViewType::e2D,
            .format = swapChainImageFormat,
            .components = { vk::ComponentSwizzle::eIdentity },
            .subresourceRange =
            {
                vk::ImageAspectFlagBits::eColor,
                0,
                1,
                0,
                1
            }

        };

        if (device.createImageView(&imageViewInfo, nullptr, &swapChainImageViews[i]) != vk::Result::eSuccess)
        {
            std::cout << "Failed to create Vulkan image view\n";
            return 0;
        }
    }


    // render pass
    vk::AttachmentDescription attachmentDescription
    {
        .format = swapChainImageFormat,
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

    if (device.createRenderPass(&renderPassInfo, nullptr, &renderPass) != vk::Result::eSuccess)
    {
        std::cout << "Failed to create render pass\n";
        return 0;
    }



    // initialisation of graphics pipeline
    auto vertexShaderCode = LoadShader("../../Shaders/Vert.spv");
    auto fragmentShaderCode = LoadShader("../../Shaders/Frag.spv");

    vk::ShaderModule vertexShaderModule = CreateShaderModule(vertexShaderCode);
    vk::ShaderModule fragmentShaderModule = CreateShaderModule(fragmentShaderCode);

    vk::PipelineShaderStageCreateInfo vertexStageInfo =
    {
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = vertexShaderModule,
        .pName = "main"
    };

    vk::PipelineShaderStageCreateInfo fragmentStageInfo =
    {
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = fragmentShaderModule,
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
        .width = static_cast<float>(swapChainExtent.width),
        .height = static_cast<float>(swapChainExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };

    vk::Rect2D scissor
    {
        .offset = { 0, 0 },
        .extent = swapChainExtent
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
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
    };


    vk::PipelineMultisampleStateCreateInfo multisampling =
    {
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = false,
        .minSampleShading = 1.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = false,
        .alphaToOneEnable = false

    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment =
    {
        .blendEnable = false,
        .srcColorBlendFactor = vk::BlendFactor::eOne,
        .dstColorBlendFactor = vk::BlendFactor::eZero,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd
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
        .pSetLayouts = nullptr,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr
    };

    if (device.createPipelineLayout(&pipelineLayoutInfo, nullptr, &pipelineLayout) != vk::Result::eSuccess)
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
        .layout = pipelineLayout,
        .renderPass = renderPass,
        .subpass = 0,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = -1
    };

    vk::Result graphicsPipelineResult = device.createGraphicsPipelines(
            nullptr,
            1,
            &graphicsPipelineCreateInfo,
            nullptr,
            &graphicsPipeline
    );
    if (graphicsPipelineResult != vk::Result::eSuccess)
    {
        std::cout << "Failed to create graphics pipeline\n";
        return 0;
    }


    device.destroyShaderModule(vertexShaderModule, nullptr);
    device.destroyShaderModule(fragmentShaderModule, nullptr);

    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (std::size_t i = 0; i < swapChainImageViews.size(); ++i)
    {
        std::array<vk::ImageView, 1> attachments = { swapChainImageViews[i] };

        vk::FramebufferCreateInfo framebufferCreateInfo =
        {
            .renderPass = renderPass,
            .attachmentCount = attachments.size(),
            .pAttachments = attachments.data(),
            .width = swapChainExtent.width,
            .height = swapChainExtent.height,
            .layers = 1
        };

        if (device.createFramebuffer(&framebufferCreateInfo, nullptr, &swapChainFramebuffers[i]) != vk::Result::eSuccess)
        {
            std::cout << "Failed to create Vulkan framebuffer\n";
            return 0;
        }
    }


    QueueFamilyIndices queueIndicesCommand = CheckQueueFamilies(physicalDevice);

    vk::CommandPoolCreateInfo commandPoolCreateInfo =
    {
        .queueFamilyIndex = queueIndicesCommand.graphicsFamily.value()
    };

    if (device.createCommandPool(&commandPoolCreateInfo, nullptr, &commandPool) != vk::Result::eSuccess)
    {
        std::cout << "Failed to created Vulkan command pool\n";
        return 0;
    }

    commandBuffers.resize(swapChainFramebuffers.size());
    vk::CommandBufferAllocateInfo cbAllocateInfo =
    {
        .commandPool = commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = static_cast<uint32_t>(commandBuffers.size())
    };

    if (device.allocateCommandBuffers(&cbAllocateInfo, commandBuffers.data()) != vk::Result::eSuccess)
    {
        std::cout << "Failed to allocate Vulkan command buffers\n";
        return 0;
    }

    for (std::size_t i = 0; i < commandBuffers.size(); ++i)
    {
        vk::CommandBufferBeginInfo cbBeginInfo = {};

        if (commandBuffers[i].begin(&cbBeginInfo) != vk::Result::eSuccess)
        {
            std::cout << "Failed to begin recording Vulkan command buffers\n";
            return 0;
        }


        vk::ClearValue clearColor(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f });

        vk::RenderPassBeginInfo renderPassBeginInfo =
        {
            .renderPass = renderPass,
            .framebuffer = swapChainFramebuffers[i],
            .renderArea = { { 0, 0 }, swapChainExtent },
            .clearValueCount = 1,
            .pClearValues = &clearColor
        };

        commandBuffers[i].beginRenderPass(&renderPassBeginInfo, vk::SubpassContents::eInline);
        commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

        commandBuffers[i].draw(3, 1, 0, 0);

        commandBuffers[i].endRenderPass();

        commandBuffers[i].end();

    }

    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

    vk::SemaphoreCreateInfo semaphoreCreateInfo = {};
    vk::FenceCreateInfo fenceCreateInfo = { .flags = vk::FenceCreateFlagBits::eSignaled };

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        if (device.createSemaphore(&semaphoreCreateInfo, nullptr, &imageAvailableSemaphores[i]) != vk::Result::eSuccess)
        {
            std::cout << "Failed to create Vulkan image semaphore\n";
            return 0;
        }


        if (device.createSemaphore(&semaphoreCreateInfo, nullptr, &renderFinishedSemaphores[i]) != vk::Result::eSuccess)
        {
            std::cout << "Failed to create Vulkan render semaphore\n";
            return 0;
        }


        if (device.createFence(&fenceCreateInfo, nullptr, &inFlightFences[i]) != vk::Result::eSuccess)
        {
            std::cout << "Failed to create Vulkan in flight fence\n";
            return 0;
        }

    }

    while (!glfwWindowShouldClose(window))
    {
        if (device.waitForFences(1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
        {
            std::cout << "Unable to wait for fence?\n";
            return 0;
        }

        uint32_t imageIndex;
        vk::Result acquireNextImageResult = device.acquireNextImageKHR(
                swapChain,
                UINT64_MAX,
                imageAvailableSemaphores[currentFrame],
                nullptr,
                &imageIndex);
        if (acquireNextImageResult != vk::Result::eSuccess)
        {
            std::cout << "Failed to acquire next image\n";
            return 0;
        }

        if (imagesInFlight[imageIndex])
            device.waitForFences(1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);


        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        vk::SubmitInfo submitInfo = {};


        std::array<vk::Semaphore, 1> waitSemaphores =
        {
            imageAvailableSemaphores[currentFrame]
        };

        std::array<vk::PipelineStageFlags, 1> waitStages =
        {
            vk::PipelineStageFlagBits::eColorAttachmentOutput
        };

        submitInfo.waitSemaphoreCount = waitSemaphores.size();
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

        std::array<vk::Semaphore, 1> signalSemaphores =
        {
            renderFinishedSemaphores[currentFrame]
        };
        submitInfo.signalSemaphoreCount = signalSemaphores.size();
        submitInfo.pSignalSemaphores = signalSemaphores.data();


        device.resetFences(1, &inFlightFences[currentFrame]);

        if (graphicsQueue.submit(1, &submitInfo, inFlightFences[currentFrame]) != vk::Result::eSuccess)
        {
            std::cout << "Failed to submit Vulkan draw command buffer\n";
            return 0;
        }

        vk::PresentInfoKHR presentInfo =
        {
            .waitSemaphoreCount = signalSemaphores.size(),
            .pWaitSemaphores = signalSemaphores.data()
        };

        std::array<vk::SwapchainKHR, 1> swapChains = { swapChain };
        presentInfo.swapchainCount = swapChains.size();
        presentInfo.pSwapchains = swapChains.data();
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        presentQueue.presentKHR(&presentInfo);


        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;


        glfwPollEvents();
    }

    vkDeviceWaitIdle(device);

    // clean up
    for (const auto& renderSemaphore : renderFinishedSemaphores)
        device.destroySemaphore(renderSemaphore, nullptr);

    for (const auto& imageSemaphore : imageAvailableSemaphores)
        device.destroySemaphore(imageSemaphore, nullptr);


    for (const auto& fence : inFlightFences)
        device.destroyFence(fence, nullptr);


    device.destroyCommandPool(commandPool, nullptr);

    for (auto framebuffer : swapChainFramebuffers)
        device.destroyFramebuffer(framebuffer, nullptr);


    for (auto imageView : swapChainImageViews)
        device.destroyImageView(imageView, nullptr);

    device.destroyPipeline(graphicsPipeline, nullptr);
    device.destroyPipelineLayout(pipelineLayout, nullptr);
    device.destroyRenderPass(renderPass, nullptr);
    device.destroySwapchainKHR(swapChain, nullptr);
    device.destroy(nullptr);
    instance.destroySurfaceKHR(surface, nullptr);
    instance.destroy(nullptr);


    glfwTerminate();

    return 0;
}
