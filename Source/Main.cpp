// todo: implement VK_LAYER_LUNARG_monitor valiation layer to display FPS
// todo: add debug message callback specifically for instance creation and deletion
/*
 * todo: swapchain images are not UniqueImage because the get images function only returns Image.
 * todo: need to find out more.
 */
// todo: maybe we should not use exceptions for Vulkan?

#include <iostream>
#include <utility>
#include <vector>
#include <optional>
#include <set>
#include <fstream>
#include <array>
#include <exception>

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

    operator bool() const
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


static VKAPI_ATTR vk::Bool32 VKAPI_CALL DebugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        vk::DebugUtilsMessageTypeFlagsEXT messageType,
        const vk::DebugUtilsMessengerCallbackDataEXT* callback,
        void* userData);
bool CheckValidationLayerSupport(const std::vector<const char*>& validationLayers);
std::vector<char> LoadShader(const std::string& path);
bool CheckDeviceExtensionSupport(const vk::PhysicalDevice& physicalDevice, const std::vector<const char*>& deviceExtensions);
QueueFamilyIndices CheckQueueFamilies(const vk::PhysicalDevice& physicalDevice, const vk::UniqueSurfaceKHR& surface);
SwapChainSupportDetails CheckSwapChainSupport(const vk::PhysicalDevice& physicalDevice, const vk::UniqueSurfaceKHR& surface);
vk::SurfaceFormatKHR SelectSwapChainSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
vk::PresentModeKHR SelectSwapChainPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
vk::Extent2D SelectSwapChainExtent(int width, int height, const vk::SurfaceCapabilitiesKHR& capabilities);
vk::UniqueShaderModule CreateShaderModule(const vk::UniqueDevice& device, const std::vector<char>& shaderCode);




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


class Application
{
public:
    Application()
    {
        if (!glfwInit())
            throw std::exception();
    }
};


class Window
{
private:
    GLFWwindow* m_Window;
    std::string m_Title;
    int m_Width, m_Height;

    bool m_FramebufferResized = false;
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

    void CreateWindow()
    {
        assert(m_Window == nullptr);

        // we will not be using OpenGL
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, true);

        m_Window = glfwCreateWindow(m_Width, m_Height, m_Title.c_str(), nullptr, nullptr);

        if (!m_Window)
            throw std::exception();

        glfwSetWindowUserPointer(m_Window, this);
        glfwSetFramebufferSizeCallback(m_Window, FramebufferResizeCallback);

    }

    [[nodiscard]] bool ShouldClose() const { return glfwWindowShouldClose(m_Window); }

    [[nodiscard]] GLFWwindow* Get() const { return m_Window; }
    [[nodiscard]] std::string GetTitle() const { return m_Title; }
    [[nodiscard]] int GetWidth() const
    {
        int width, height;
        glfwGetFramebufferSize(m_Window, &width, &height);

        return width;
    }

    [[nodiscard]] int GetHeight() const
    {
        int width = 0, height = 0;
        glfwGetFramebufferSize(m_Window, &width, &height);

        return height;
    }

    bool& IsFramebufferResized() { return m_FramebufferResized; }
private:
    static void FramebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        auto win = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
        win->m_FramebufferResized = true;
    }
};


struct VulkanRendererInfo
{
    std::string applicationName;
    uint32_t applicationVersion;

    std::string engineName;
    uint32_t engineVersion;

    uint32_t rendererVersion;
};


class VulkanRenderer
{
private:
    Window& m_Window;

    VulkanRendererInfo m_RendererInfo;

    vk::UniqueInstance m_Instance;

#ifndef NDEBUG
    vk::DispatchLoaderDynamic m_DynamicLoader;
    vk::UniqueHandle<vk::DebugUtilsMessengerEXT, vk::DispatchLoaderDynamic> m_Messenger;
#endif

    vk::UniqueSurfaceKHR m_Surface;

    vk::PhysicalDevice m_PhysicalDevice;
    QueueFamilyIndices m_QueueFamilies;

    vk::UniqueDevice m_Device;
    vk::Queue m_Graphics, m_Present;

    SwapChainSupportDetails m_SwapchainSupport;
    vk::SurfaceFormatKHR m_SwapchainSurfaceFormat;
    vk::PresentModeKHR m_SwapchainPresentMode;
    vk::Extent2D m_SwapchainExtent;
    vk::UniqueSwapchainKHR m_Swapchain;
    std::vector<vk::UniqueImageView> m_SwapchainImageViews;
    std::vector<vk::Image> m_SwapchainImages;
    vk::Format m_SwapchainImageFormat;

    vk::UniqueRenderPass m_RenderPass;
    vk::UniquePipelineLayout m_PipelineLayout;

    std::vector<vk::DynamicState> m_DynamicStates;

    // todo: find out why this is a vector
    std::vector<vk::UniquePipeline> m_Pipeline;
    std::vector<vk::UniqueFramebuffer> m_Framebuffers;
    vk::UniqueCommandPool m_CommandPool;
    std::vector<vk::UniqueCommandBuffer> m_CommandBuffers;


    const int MAX_FRAMES_IN_FLIGHT = 2;
    std::size_t m_CurrentFrame = 0;

    std::vector<vk::UniqueSemaphore> m_ImageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> m_RenderFinishedSemaphores;
    std::vector<vk::UniqueFence> m_InFlightFences;

    std::vector<vk::UniqueFence*> m_ImagesInFlight;
public:
    // todo: find out what explicit does here
    explicit VulkanRenderer(VulkanRendererInfo vulkanRendererInfo, Window& window)
        : m_RendererInfo(std::move(vulkanRendererInfo)), m_Window(window)
    {

    }

#ifndef NDEBUG
    void CreateDebugMessenger()
    {
        using SeverityFlagBit = vk::DebugUtilsMessageSeverityFlagBitsEXT;
        using TypeFlagBit = vk::DebugUtilsMessageTypeFlagBitsEXT;

        // note: TypeFlagBit::eGeneral was used here before
        vk::DebugUtilsMessengerCreateInfoEXT debugInfo
        {
            .messageSeverity = SeverityFlagBit::eVerbose | SeverityFlagBit::eWarning | SeverityFlagBit::eError,
            .messageType = TypeFlagBit::eValidation | TypeFlagBit::ePerformance,
            .pfnUserCallback = reinterpret_cast<PFN_vkDebugUtilsMessengerCallbackEXT>(DebugCallback),
            .pUserData = nullptr
        };

        m_DynamicLoader = vk::DispatchLoaderDynamic(m_Instance.get(), vkGetInstanceProcAddr);
        m_Messenger = m_Instance->createDebugUtilsMessengerEXTUnique(debugInfo, nullptr, m_DynamicLoader);
    }
#endif

    void CreateInstance()
    {
        assert(!m_Instance);

        vk::ApplicationInfo applicationInfo
        {
            .pApplicationName = m_RendererInfo.applicationName.c_str(),
            .applicationVersion = m_RendererInfo.applicationVersion,
            .pEngineName = m_RendererInfo.engineName.c_str(),
            .engineVersion = m_RendererInfo.engineVersion,
            .apiVersion = m_RendererInfo.rendererVersion
        };

        vk::InstanceCreateInfo instanceInfo
        {
            .pApplicationInfo = &applicationInfo,
            .enabledLayerCount = static_cast<uint32_t>(validationLayers.size()),
            .ppEnabledLayerNames = validationLayers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size()),
            .ppEnabledExtensionNames = instanceExtensions.data()
        };

        m_Instance = vk::createInstanceUnique(instanceInfo);
    }

    void CreateSurface()
    {
       // assert(!m_Surface);

        if (glfwCreateWindowSurface(m_Instance.get(), m_Window.Get(), nullptr, reinterpret_cast<VkSurfaceKHR*>(&m_Surface.get())) != VK_SUCCESS)
        {
            std::cout << "Failed to create Vulkan window surface\n";
            throw std::exception();
        }

        m_Surface = vk::UniqueSurfaceKHR(m_Surface.get(), m_Instance.get());
    }

    void SelectPhysicalDevice()
    {
        //assert(!m_PhysicalDevice);

        // get the total number of GPU's that support Vulkan on the system
        std::vector<vk::PhysicalDevice> physicalDevices = m_Instance->enumeratePhysicalDevices();
        if (physicalDevices.empty())
        {
            std::cout << "No GPU's found with Vulkan support\n";
            throw std::exception();
        }

        // todo: for simplicity we will just use the first GPU found
        m_PhysicalDevice = physicalDevices[0];

        // check if physical device supports required device extensions
        if (!IsRequiredDeviceExtensionsFound(m_PhysicalDevice))
        {
            std::cout << "GPU does not support required device extensions\n";
            throw std::exception();
        }

        // check if physical device supports required queue families
        m_QueueFamilies = CheckQueueFamilies(m_PhysicalDevice);
        if (!m_QueueFamilies)
        {
            std::cout << "GPU does not support required queue families\n";
            throw std::exception();
        }

        // check if the physical device support the swpachain formats // todo: more info
        m_SwapchainSupport = CheckSwapChainSupport(m_PhysicalDevice);
        bool swapChainAdequate = !m_SwapchainSupport.formats.empty() && !m_SwapchainSupport.presentModes.empty();

        if (!swapChainAdequate)
        {
            std::cout << "GPU does not support swapchain format\n";
            throw std::exception();
        }

        if (!m_PhysicalDevice)
        {
            std::cout << "No GPU's found with required Vulkan features\n";
            throw std::exception();
        }
    }

    void CreateDevice()
    {
        //assert(!m_Device);

        std::vector<vk::DeviceQueueCreateInfo> deviceQueueInfos;

        // todo: why does using an array here cause an error when not when using a set?
        std::set<uint32_t> uniqueQueueFamilies =
        {
            m_QueueFamilies.graphicsFamily.value(),
            m_QueueFamilies.presentFamily.value()
        };

        float queuePriority = 1.0f;
        for (const auto& queueFamily : uniqueQueueFamilies)
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

        m_Device = m_PhysicalDevice.createDeviceUnique(deviceInfo);

        // Fill queues
        m_Device->getQueue(m_QueueFamilies.graphicsFamily.value(), 0, &m_Graphics);
        m_Device->getQueue(m_QueueFamilies.presentFamily.value(), 0, &m_Present);
    }

    void CreateSwapchain()
    {
        //assert(!m_Swapchain);

        m_SwapchainSurfaceFormat = SelectSwapChainSurfaceFormat(m_SwapchainSupport.formats);
        m_SwapchainPresentMode = SelectSwapChainPresentMode(m_SwapchainSupport.presentModes);
        m_SwapchainExtent = SelectSwapChainExtent(m_Window.GetWidth(), m_Window.GetHeight(), m_SwapchainSupport.capabilities);

        uint32_t imageCount = m_SwapchainSupport.capabilities.minImageCount + 1;
        if (m_SwapchainSupport.capabilities.maxImageCount > 0 && imageCount > m_SwapchainSupport.capabilities.maxImageCount)
            imageCount = m_SwapchainSupport.capabilities.maxImageCount;


        std::array<uint32_t, 2> queueFamilyIndices =
        {
            m_QueueFamilies.graphicsFamily.value(),
            m_QueueFamilies.presentFamily.value()
        };


        vk::SharingMode imageSharingMode;
        uint32_t queueFamilyIndexCount;
        uint32_t* pQueueFamilyIndices;

        if (m_QueueFamilies.graphicsFamily != m_QueueFamilies.presentFamily)
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
            .surface = m_Surface.get(),
            .minImageCount = imageCount,
            .imageFormat = m_SwapchainSurfaceFormat.format,
            .imageColorSpace = m_SwapchainSurfaceFormat.colorSpace,
            .imageExtent = m_SwapchainExtent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = imageSharingMode,
            .queueFamilyIndexCount = queueFamilyIndexCount,
            .pQueueFamilyIndices = pQueueFamilyIndices,
            .preTransform = m_SwapchainSupport.capabilities.currentTransform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = m_SwapchainPresentMode,
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE
        };

        m_Swapchain = m_Device->createSwapchainKHRUnique(swapchainInfo);
    }

    void CreateSwapchainImageViews()
    {
        //assert(m_SwapchainImages.empty());

        m_SwapchainImages = m_Device->getSwapchainImagesKHR(m_Swapchain.get());
        m_SwapchainImageFormat = m_SwapchainSurfaceFormat.format;

        for (auto& swapchainImage : m_SwapchainImages)
        {
            vk::ImageViewCreateInfo imageViewInfo
            {
                .image = swapchainImage,
                .viewType = vk::ImageViewType::e2D,
                .format = m_SwapchainImageFormat,
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

            m_SwapchainImageViews.push_back(m_Device->createImageViewUnique(imageViewInfo));
        }
    }

    void CreateRenderPass()
    {
        //assert(!m_RenderPass);

        vk::AttachmentDescription attachmentDescription
        {
            .format = m_SwapchainImageFormat,
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

        m_RenderPass = m_Device->createRenderPassUnique(renderPassInfo);
    }

    void CreateGraphicsPipelineLayout()
    {
        //assert(!m_PipelineLayout);

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo =
        {
            .setLayoutCount = 0,
            .pushConstantRangeCount = 0,
        };

        m_PipelineLayout = m_Device->createPipelineLayoutUnique(pipelineLayoutInfo);
    }

    void CreateGraphicsPipeline()
    {
        //assert(m_Pipeline.empty());

        std::vector<char> vertexShaderCode;
        std::vector<char> fragmentShaderCode;
        vertexShaderCode = LoadShader("../Shaders/Vert.spv");
        fragmentShaderCode = LoadShader("../Shaders/Frag.spv");

        vk::UniqueShaderModule vertexShaderModule = CreateShaderModule(m_Device, vertexShaderCode);
        vk::UniqueShaderModule fragmentShaderModule = CreateShaderModule(m_Device, fragmentShaderCode);

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

        vk::PipelineViewportStateCreateInfo viewportState =
        {
            .viewportCount = 1,
            .pViewports = nullptr,
            .scissorCount = 1,
            .pScissors = nullptr
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

        m_DynamicStates =
        {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };

        vk::PipelineDynamicStateCreateInfo dynamicStateInfo =
        {
            .dynamicStateCount = static_cast<uint32_t>(m_DynamicStates.size()),
            .pDynamicStates = m_DynamicStates.data()
        };

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
            .pDynamicState = &dynamicStateInfo,
            .layout = m_PipelineLayout.get(),
            .renderPass = m_RenderPass.get(),
            .subpass = 0,
            .basePipelineHandle = nullptr,
        };

        // todo: find out why this returns a list of graphics pipelines
        m_Pipeline = m_Device->createGraphicsPipelinesUnique(nullptr, graphicsPipelineCreateInfo, nullptr).value;
    }

    void CreateFramebuffers()
    {
        //assert(m_Framebuffers.empty());

        for (auto& swapchainImage : m_SwapchainImageViews)
        {
            std::array<vk::ImageView, 1> attachments = { swapchainImage.get() };
            vk::FramebufferCreateInfo framebufferCreateInfo =
            {
                .renderPass = m_RenderPass.get(),
                .attachmentCount = attachments.size(),
                .pAttachments = attachments.data(),
                .width = m_SwapchainExtent.width,
                .height = m_SwapchainExtent.height,
                .layers = 1
            };

            m_Framebuffers.push_back(m_Device->createFramebufferUnique(framebufferCreateInfo, nullptr));
        }
    }

    void CreateCommandPool()
    {
        // todo: no need to destory command pool. Double check
        //assert(!m_CommandPool);

        vk::CommandPoolCreateInfo commandPoolCreateInfo =
        {
            .queueFamilyIndex = m_QueueFamilies.graphicsFamily.value()
        };

        m_CommandPool = m_Device->createCommandPoolUnique(commandPoolCreateInfo, nullptr);
    }

    void AllocateCommandBuffers()
    {
        //assert(m_CommandBuffers.empty());

        vk::CommandBufferAllocateInfo cbAllocateInfo =
        {
            .commandPool = m_CommandPool.get(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = static_cast<uint32_t>(m_Framebuffers.size())
        };

        m_CommandBuffers = m_Device->allocateCommandBuffersUnique(cbAllocateInfo);
    }

    void RecordCommandBuffer()
    {
        //assert(!m_CommandBuffers.empty());

        int index = 0;
        for (auto& commandBuffer : m_CommandBuffers)
        {
            vk::CommandBufferBeginInfo cbBeginInfo = {};

            if (commandBuffer->begin(&cbBeginInfo) != vk::Result::eSuccess)
            {
                std::cout << "Failed to begin recording Vulkan command buffers\n";
                throw std::exception();
            }

            vk::ClearValue clearColor(std::array<float, 4>({ 0.04f, 0.04f, 0.04f, 1.0f }));

            vk::RenderPassBeginInfo renderPassBeginInfo =
            {
                .renderPass = m_RenderPass.get(),
                .framebuffer = m_Framebuffers[index].get(),
                .renderArea = {{ 0, 0 }, m_SwapchainExtent },
                .clearValueCount = 1,
                .pClearValues = &clearColor
            };





            commandBuffer->beginRenderPass(&renderPassBeginInfo, vk::SubpassContents::eInline);

            vk::Viewport viewport =
            {
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(m_SwapchainExtent.width),
                .height = static_cast<float>(m_SwapchainExtent.height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f
            };

            vk::Rect2D scissor
            {
                .offset = { 0, 0 },
                .extent = m_SwapchainExtent
            };

            commandBuffer->setViewport(0, 1, &viewport);
            commandBuffer->setScissor(0, 1, &scissor);

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, m_Pipeline[0].get());

            commandBuffer->draw(3, 1, 0, 0);

            commandBuffer->endRenderPass();

            commandBuffer->end();

            ++index;
        }
    }

    void CreateSemaphores()
    {
        //assert(m_ImageAvailableSemaphores.empty() && m_RenderFinishedSemaphores.empty());

        vk::SemaphoreCreateInfo semaphoreCreateInfo = {};
        for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            m_ImageAvailableSemaphores.push_back(m_Device->createSemaphoreUnique(semaphoreCreateInfo, nullptr));
            m_RenderFinishedSemaphores.push_back(m_Device->createSemaphoreUnique(semaphoreCreateInfo, nullptr));
        }
    }

    void CreateFences()
    {
        //assert(m_InFlightFences.empty());

        vk::FenceCreateInfo fenceCreateInfo =
        {
            .flags = vk::FenceCreateFlagBits::eSignaled
        };

        for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            m_InFlightFences.push_back(m_Device->createFenceUnique(fenceCreateInfo, nullptr));
        }


        // todo: this is temp
        m_ImagesInFlight.resize(m_SwapchainImages.size());
    }

    void WaitForFence()
    {
        if (m_Device->waitForFences(1, &m_InFlightFences[m_CurrentFrame].get(), VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
        {
            std::cout << "Unable to wait for fence?\n";
            throw std::exception();
        }
    }

    void AcquireNextImage()
    {
        auto [acquireImageResult, imageIndex] = m_Device->acquireNextImageKHR(m_Swapchain.get(), UINT64_MAX, m_ImageAvailableSemaphores[m_CurrentFrame].get(), nullptr);

        switch (acquireImageResult)
        {
            case vk::Result::eErrorOutOfDateKHR:
            {
                RecreateSwapchain();
            } break;
            case vk::Result::eSuboptimalKHR:
            {
                // not the best swapchain case however, presentation engine is still able to render
            } break;
            case vk::Result::eErrorSurfaceLostKHR:
            {
                // todo: will need to destroy and then create a new surface
                // todo: might need to also recreate the swapchain
            } break;
            default:
                break;
        }

        if (m_ImagesInFlight[imageIndex])
        {

            if (m_Device->waitForFences(m_ImagesInFlight[imageIndex]->get(), VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
            {
                std::cout << "Error for 'waitForFences'.\n";
                throw std::exception();
            }

        }

        m_ImagesInFlight[imageIndex] = &m_InFlightFences[m_CurrentFrame];
        
        
        TempFunction(imageIndex);
    }
    
    void TempFunction(uint32_t imageIndex)
    {
        vk::SubmitInfo submitInfo = {};
        std::array<vk::Semaphore, 1> waitSemaphores =
        {
            m_ImageAvailableSemaphores[m_CurrentFrame].get()
        };

        std::array<vk::PipelineStageFlags, 1> waitStages =
        {
            vk::PipelineStageFlagBits::eColorAttachmentOutput
        };

        submitInfo.waitSemaphoreCount = waitSemaphores.size();
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_CommandBuffers[imageIndex].get();

        std::array<vk::Semaphore, 1> signalSemaphores =
        {
            m_RenderFinishedSemaphores[m_CurrentFrame].get()
        };
        submitInfo.signalSemaphoreCount = signalSemaphores.size();
        submitInfo.pSignalSemaphores = signalSemaphores.data();

        m_Device->resetFences(m_InFlightFences[m_CurrentFrame].get());

        if (m_Graphics.submit(1, &submitInfo, m_InFlightFences[m_CurrentFrame].get()) != vk::Result::eSuccess)
        {
            std::cout << "Failed to submit Vulkan draw command buffer\n";
            throw std::exception();
        }

        vk::PresentInfoKHR presentInfo =
        {
            .waitSemaphoreCount = signalSemaphores.size(),
            .pWaitSemaphores = signalSemaphores.data()
        };

        std::array<vk::SwapchainKHR, 1> swapChains = { m_Swapchain.get() };
        presentInfo.swapchainCount = swapChains.size();
        presentInfo.pSwapchains = swapChains.data();
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        vk::Result presentResult = m_Present.presentKHR(&presentInfo);
        switch (presentResult)
        {
            case vk::Result::eErrorOutOfDateKHR:
            case vk::Result::eSuboptimalKHR:
            {
                RecreateSwapchain();
            } break;
            case vk::Result::eErrorSurfaceLostKHR:
            {
                // todo: will need to destroy and then create a new surface
                // todo: might need to also recreate the swapchain
            } break;
            default:
                break;
        }
        if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR)
        {
            std::cout << "present recreate\n";
            m_Window.IsFramebufferResized() = false;
            RecreateSwapchain();

        }
        else if (presentResult != vk::Result::eSuccess)
        {
            std::cout << "Unable to present\n";
            throw std::exception();
        }

        m_CurrentFrame = (m_CurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    }
    
    void DeviceWaitIdle()
    {
        m_Device->waitIdle();
    }

    void CleanUpSwapchain()
    {
        for (auto& framebuffer : m_Framebuffers)
            framebuffer.release();

        for (auto& commandBuffer : m_CommandBuffers)
            commandBuffer.release();

        m_Pipeline[0].release();

        m_PipelineLayout.release();
        m_RenderPass.release();

        for (auto& imageView : m_SwapchainImageViews)
            imageView.release();

        m_Swapchain.release();


        // todo: clear lists
        m_Framebuffers.clear();
        m_CommandBuffers.clear();
        m_SwapchainImageViews.clear();
    }

    void RecreateSwapchain()
    {
        //std::cout << "Window resized and swapchain should be recreated\n";
        // todo: this is meant for minimisation but we get stuck in a loop which results in a big spike in usage
       /* while (m_Window.GetWidth() == 0 || m_Window.GetHeight() == 0)
            glfwWaitEvents();*/

        m_Device->waitIdle();

        CleanUpSwapchain();

        CreateSwapchain();
        CreateSwapchainImageViews();
        CreateRenderPass();
        CreateGraphicsPipelineLayout(); // todo: check if this is actually used
        CreateGraphicsPipeline();
        CreateFramebuffers();
        CreateCommandPool();
        AllocateCommandBuffers();
        RecordCommandBuffer();
    }


private:
    [[nodiscard]] QueueFamilyIndices CheckQueueFamilies(const vk::PhysicalDevice& physicalDevice)
    {
        std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();

        QueueFamilyIndices indices = {};
        vk::Bool32 presentSupported = false;
        for (std::size_t i = 0; i < queueFamilies.size(); ++i)
        {
            if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics)
                indices.graphicsFamily = i;

            if (physicalDevice.getSurfaceSupportKHR(i, m_Surface.get(), &presentSupported) != vk::Result::eSuccess)
                continue;

            if (presentSupported)
                indices.presentFamily = i;

            if (indices)
                break;
        }

        return indices;
    }

    [[nodiscard]] static bool IsRequiredDeviceExtensionsFound(const vk::PhysicalDevice& physicalDevice)
    {
        std::vector<vk::ExtensionProperties> availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

        for (const auto& requiredExtension : deviceExtensions)
        {
            const auto extensionFound = [&requiredExtension](vk::ExtensionProperties& deviceExtension)
            {
                return requiredExtension == static_cast<std::string>(deviceExtension.extensionName);
            };

            const auto iter = std::find_if(availableExtensions.begin(), availableExtensions.end(), extensionFound);
            if (iter == availableExtensions.end())
                return false;
        }

        return true;
    }

    [[nodiscard]] SwapChainSupportDetails CheckSwapChainSupport(const vk::PhysicalDevice& physicalDevice)
    {
        SwapChainSupportDetails details =
        {
            .capabilities = physicalDevice.getSurfaceCapabilitiesKHR(m_Surface.get()),
            .formats = physicalDevice.getSurfaceFormatsKHR(m_Surface.get()),
            .presentModes = physicalDevice.getSurfacePresentModesKHR(m_Surface.get())
        };

        return details;
    }

};

int main()
{
    Application application;

    Window window("Vulkan Application", 400, 400);
    window.CreateWindow();

    VulkanRendererInfo rendererInfo
    {
        .applicationName = window.GetTitle(),
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .engineName = "Internal Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .rendererVersion = VK_API_VERSION_1_0
    };

    VulkanRenderer vulkanRenderer(rendererInfo, window);

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    const std::vector<const char*> glfwExtensionsList(glfwExtensions, glfwExtensions + glfwExtensionCount);
    instanceExtensions.insert(instanceExtensions.end(), glfwExtensionsList.begin(), glfwExtensionsList.end());

    vulkanRenderer.CreateInstance();
#ifndef NDEBUG
    vulkanRenderer.CreateDebugMessenger();
    
    if (!CheckValidationLayerSupport(validationLayers))
    {
        std::cout << "Validation layers requested but not available\n";
        return 0;
    }
#endif

    vulkanRenderer.CreateSurface();
    vulkanRenderer.SelectPhysicalDevice();
    vulkanRenderer.CreateDevice();
    vulkanRenderer.CreateSwapchain();
    vulkanRenderer.CreateSwapchainImageViews();
    vulkanRenderer.CreateRenderPass();
    vulkanRenderer.CreateGraphicsPipelineLayout();
    vulkanRenderer.CreateGraphicsPipeline();
    vulkanRenderer.CreateFramebuffers();
    vulkanRenderer.CreateCommandPool();
    vulkanRenderer.AllocateCommandBuffers();
    vulkanRenderer.RecordCommandBuffer();
    vulkanRenderer.CreateSemaphores();
    vulkanRenderer.CreateFences();

    while (!window.ShouldClose())
    {
        vulkanRenderer.WaitForFence();

        vulkanRenderer.AcquireNextImage();

        glfwPollEvents();
    }

    vulkanRenderer.DeviceWaitIdle();


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


std::vector<char> LoadShader(const std::string& path)
{
    std::ifstream shader(path, std::ios::ate | std::ios::binary);

    if (!shader.is_open())
        throw std::exception();

    size_t fileSize = (size_t) shader.tellg();
    std::vector<char> buffer(fileSize);
    shader.seekg(0);
    shader.read(buffer.data(), fileSize);

    return buffer;
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
