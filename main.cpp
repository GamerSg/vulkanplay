#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>


GLFWwindow* window;
VkSurfaceKHR surface;                                       // Surface used by window to render to
vk::Instance instance;
vk::DispatchLoaderDynamic dldy;                             //Dynamic Loader for extensions
VkDebugUtilsMessengerEXT debugMessenger;
vk::PhysicalDevice device;
vk::Device gpu;
vk::PresentModeKHR presMode;
vk::Rect2D scissors;
vk::Queue graphicsQueue;

unsigned int queueIndex, presentQueue;
unsigned int FPS;
double lastFrame;

vk::SwapchainKHR swapChain;
std::vector<vk::Image> swapImgs;
std::vector<vk::ImageView> swapImgViews;
std::vector<vk::Framebuffer> frameBuffers;
vk::Buffer VBO;

vk::RenderPass renderPass;
vk::PipelineLayout pipelineLayout;
vk::SurfaceFormatKHR sf;

vk::Pipeline graphicsPipeline;

vk::CommandPool commandPool;
std::vector<vk::CommandBuffer> commandBuffers;

std::vector<vk::Semaphore> imgAvlblSMPH, renderFinishSMPH;
std::vector<vk::Fence> cpuSyncFN;
size_t currentFrame = 0;

const int WIDTH = 800;
const int HEIGHT = 600;
vk::Extent2D actualExtent;                                  // Extent used currently
const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"
  };
const std::vector<const char*> devEXTsNeeded = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
  };

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback( VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) 
{

    std::cerr << "[VULKAN]:" << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;
    
    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex);     

        return bindingDescription;
    }
    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
    std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions = {};
    
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);
    
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    return attributeDescriptions;
    }
};

const std::vector<Vertex> vertices = {
    {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
};

std::vector<char> loadFileToMem(const std::string& filename)
{
    std::ifstream fin(filename, std::ios::ate | std::ios::binary);
    std::vector<char> mem;
    if (!fin.is_open())
    {
        std::cerr << "Unable to load file " << filename << std::endl; 
        return mem;
    }
    
    size_t fileSize = (size_t) fin.tellg();
    mem.resize(fileSize);
    // Go back to start & read the file
    fin.seekg(0);   
    fin.read(mem.data(), fileSize);
    
    fin.close();
    return mem;    
}
  
vk::ShaderModule createShaderModule(const std::vector<char>& data)
{
    vk::ShaderModuleCreateInfo sm(vk::ShaderModuleCreateFlags(), data.size(), reinterpret_cast<const uint32_t*>(data.data()));
    return gpu.createShaderModule(sm);
}

void createVertexBuffer()
{
    vk::BufferCreateInfo bci(vk::BufferCreateFlags(), sizeof(vertices[0]) * vertices.size(), vk::BufferUsageFlagBits::eVertexBuffer, vk::SharingMode::eExclusive);
    VBO = gpu.createBuffer(bci);
    
    
}
  
void createPipeline()
{
    auto VS = loadFileToMem("./shaders/vert.spv");
    auto FS = loadFileToMem("./shaders/frag.spv");
    
    vk::ShaderModule vsm = createShaderModule(VS);
    vk::ShaderModule fsm = createShaderModule(FS);
    
    vk::PipelineShaderStageCreateInfo vsStage(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, vsm, "main");
    vk::PipelineShaderStageCreateInfo fsStage(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, fsm, "main");
    vk::PipelineShaderStageCreateInfo stages[] = {vsStage, fsStage};
    
    vk::PipelineVertexInputStateCreateInfo visci;    
    auto vertexBindingDesc = Vertex::getBindingDescription();
    auto vertexInputAttrDesc = Vertex::getAttributeDescriptions();
    visci.setVertexBindingDescriptionCount(1);
    visci.setPVertexBindingDescriptions(&vertexBindingDesc);
    visci.setVertexAttributeDescriptionCount(2);
    visci.setPVertexAttributeDescriptions(&vertexInputAttrDesc[0]);
    
    // VBO primitive type (almost always triangle list)
    vk::PipelineInputAssemblyStateCreateInfo iasci(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::eTriangleList);
    // Viewport
    vk::Viewport viewport(0, 0, actualExtent.width, actualExtent.height,  0.0f,  1.0f);
    
    scissors.offset.x = 0;
    scissors.offset.y = 0;
    scissors.extent.width = actualExtent.width;
    scissors.extent.height = actualExtent.height;
    vk::PipelineViewportStateCreateInfo vsci(vk::PipelineViewportStateCreateFlags(), 1, &viewport, 1, &scissors);
    // Rasterization (Back face culling,  front face triangle order, fill mode or wireframe, etc..)
    vk::PipelineRasterizationStateCreateInfo rsci(vk::PipelineRasterizationStateCreateFlags(), 0, 0, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack,vk::FrontFace::eClockwise, 0, 0, 0, 0, 1.0f);
    // MSAA
    vk::PipelineMultisampleStateCreateInfo msci;
    // Blending
    vk::PipelineColorBlendAttachmentState cbas(1, vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR |  vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA );
    vk::PipelineColorBlendStateCreateInfo sbsci(vk::PipelineColorBlendStateCreateFlags(), 0, vk::LogicOp::eCopy, 1, &cbas);
    // For setting uniform/constant variables for shaders
    pipelineLayout = gpu.createPipelineLayout(vk::PipelineLayoutCreateInfo());
    // Finally create pipeline and link it all together
    vk::GraphicsPipelineCreateInfo gpci;
    gpci.setStageCount(2);
    gpci.setPStages(stages);
    gpci.setPVertexInputState(&visci);
    gpci.setPInputAssemblyState(&iasci);
    gpci.setPViewportState(&vsci);
    gpci.setPRasterizationState(&rsci);
    gpci.setPMultisampleState(&msci);
    gpci.setPColorBlendState(&sbsci);
    gpci.setLayout(pipelineLayout);
    gpci.setRenderPass(renderPass);
    gpci.setSubpass(0);
    graphicsPipeline = gpu.createGraphicsPipeline(nullptr, gpci);
    
    
    
    gpu.destroy(vsm);
    gpu.destroy(fsm);
   
}

void createRenderPass()
{
    vk::AttachmentDescription ad(vk::AttachmentDescriptionFlags(), sf.format, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);
    vk::AttachmentReference ar(0, vk::ImageLayout::eColorAttachmentOptimal);
    vk::SubpassDescription sp(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics);
    sp.setColorAttachmentCount(1);
    sp.setPColorAttachments(&ar);
    
    vk::RenderPassCreateInfo rpci(vk::RenderPassCreateFlags(), 1, &ad, 1, &sp);
    
    vk::SubpassDependency spd(VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eColorAttachmentRead |  vk::AccessFlagBits::eColorAttachmentWrite);
    rpci.setDependencyCount(1);
    rpci.setPDependencies(&spd);
    
    renderPass = gpu.createRenderPass(rpci);
    
    
    
}

void createFrameBuffers()
{
    frameBuffers.resize(swapImgViews.size());
    for (int i = 0; i < swapImgViews.size(); ++i)
    {
        vk::ImageView attch[] = {
            swapImgViews[i]
        };
        vk::FramebufferCreateInfo fbci(vk::FramebufferCreateFlags(), renderPass, 1, attch, actualExtent.width, actualExtent.height, 1);
        frameBuffers[i] = gpu.createFramebuffer(fbci);
        
    }
}

void createSwapChain()
{
    // Create Swapchain
         std::cout << "Surface props" << std::endl;
        std::cout << "-----------------" << std::endl;

        auto surfCap = device.getSurfaceCapabilitiesKHR(surface);
        auto surfFormats = device.getSurfaceFormatsKHR(surface);
        auto surfPresModes = device.getSurfacePresentModesKHR(surface);        

        std::cout<< "Surface Cap : " << surfCap.maxImageExtent.width << "X" << surfCap.maxImageExtent.height << std::endl;
        std::cout << "Surface images " << surfCap.minImageCount << " - " << surfCap.maxImageCount << std::endl;
        if (surfFormats.size() ==  1 && static_cast<VkFormat>(surfFormats[0].format) == VK_FORMAT_UNDEFINED)
        {
            sf.format = vk::Format::eB8G8R8A8Unorm; 
            sf.colorSpace = vk::ColorSpaceKHR::eVkColorspaceSrgbNonlinear;
        // Use default
        }
        else
        {
            for (auto& fmt :  surfFormats)
            {
                std::cout<< static_cast<unsigned int>(fmt.format) << " " <<  static_cast<unsigned int>(fmt.colorSpace) << std::endl;
            }
            sf = surfFormats[0];
        }
        std::cout << "VSync Modes\n";
        for (auto& md :  surfPresModes)
        {
            std::cout <<  static_cast<unsigned int>(md) << std::endl;
            if (md ==  vk::PresentModeKHR::eImmediate)
            {
                presMode = vk::PresentModeKHR::eImmediate;                
                break;
            }
            presMode = md;            
        }

        vk::SwapchainCreateInfoKHR sci(vk::SwapchainCreateFlagsKHR(), surface, surfCap.minImageCount + 1, sf.format, sf.colorSpace,surfCap.currentExtent, 1, vk::ImageUsageFlags(vk::ImageUsageFlagBits::eColorAttachment), vk::SharingMode::eExclusive, 0, nullptr, surfCap.currentTransform, vk::CompositeAlphaFlagBitsKHR::eOpaque, presMode, VK_TRUE, nullptr); 
        actualExtent = surfCap.currentExtent;
        swapChain = gpu.createSwapchainKHR(sci);
        swapImgs = gpu.getSwapchainImagesKHR(swapChain);
        std::cout << "Swap images size : " << swapImgs.size() << std::endl;
        for (auto & i :  swapImgs)
        {
            std::cout << "Swap Img : " << i << std::endl;
        }
        graphicsQueue = gpu.getQueue(0, 0);
        // Create ImageViews
        swapImgViews.resize(swapImgs.size());
        for ( int i = 0; i < swapImgs.size(); ++i)
        {
            swapImgs[i];
            vk::ImageViewCreateInfo ivci;
            ivci.flags = vk::ImageViewCreateFlags();
            ivci.image = swapImgs[i];
            ivci.viewType = vk::ImageViewType::e2D;
            ivci.format = sf.format;
            ivci.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            ivci.subresourceRange.levelCount = 1;
            ivci.subresourceRange.layerCount = 1;
            swapImgViews[i] = gpu.createImageView(ivci);
        }
        
}

void cleanUpSwapChain()
{
    for (auto & fb :  frameBuffers) 
    {
        gpu.destroy(fb);
    }
    for (auto &iv :  swapImgViews)
    {
        gpu.destroy(iv);
    }     
    gpu.destroy(renderPass);
    gpu.destroy(graphicsPipeline);
    gpu.destroy(pipelineLayout);
    gpu.destroy(swapChain);

}


void createCommandBuffers()
{ 
    vk::CommandBufferAllocateInfo cbai(commandPool, vk::CommandBufferLevel::ePrimary, frameBuffers.size());
    commandBuffers = gpu.allocateCommandBuffers(cbai);
    for (int i = 0; i < commandBuffers.size(); ++i)
    {
        vk::CommandBufferBeginInfo cbbi(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
        commandBuffers[i].begin(cbbi);                      // Begin Recording
        std::array<float, 4> bgColor = {0.0f, 0.0f, 0.0f, 0.4f};        
        vk::ClearValue cv(bgColor);
        
        vk::RenderPassBeginInfo rpbi(renderPass, frameBuffers[i], scissors, 1, &cv);
        commandBuffers[i].beginRenderPass(rpbi, vk::SubpassContents::eInline);          // Begin Render Pass        
        //Bind our painstakingly created pipeline
        commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline); 
        commandBuffers[i].draw(3, 1, 0, 0);
        commandBuffers[i].endRenderPass();
        commandBuffers[i].end();
        
    }
    
    
}

void recreateSwapChain()
{
    gpu.waitIdle();
    
    cleanUpSwapChain();
    
    createSwapChain();
    createRenderPass();
    createPipeline();
    createFrameBuffers();
    createCommandBuffers();    
}


void VulkanStuff()
{
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;

    //Load required extensions
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> loadExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    std::cout<<"Extensions Required : "<<glfwExtensions<<std::endl;
    auto appInfo = vk::ApplicationInfo("VulkanPlay",VK_MAKE_VERSION(1, 0, 0),"SYPX Engine",VK_MAKE_VERSION(1, 0, 0),VK_API_VERSION_1_0 );

    try
    {
        if (enableValidationLayers) 
        {      

            //Enable debug extension
            loadExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);        
            std::cout<<"Enabling Validation Layers...\n";    
            instance = vk::createInstance(vk::InstanceCreateInfo( vk::InstanceCreateFlags(), &appInfo, validationLayers.size(),validationLayers.data(),loadExtensions.size(), loadExtensions.data()));         

            dldy.init(instance);

            vk::DebugUtilsMessengerCreateInfoEXT debugCB;
            debugCB.messageSeverity =
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose;
            debugCB.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;
            //Register call back for Vulkan Debugging        
            debugCB.pfnUserCallback = debugCallback;        
            debugMessenger = instance.createDebugUtilsMessengerEXT(debugCB, nullptr, dldy);
        }

        else
        {
            instance = vk::createInstance(vk::InstanceCreateInfo( vk::InstanceCreateFlags(), &appInfo, 0,nullptr,glfwExtensionCount,glfwExtensions ));
        }

        if (!instance)
        {
            std::cout << "Vulkan not supported,  exiting..." << std::endl;
        }
        // Create Window Surface
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
          }
        std::cout << "=======================" << std::endl;
        std::cout << "Instance Extensions" << std::endl;
        std::cout << "=======================" << std::endl;
        //Extensions supported
        auto extensions = vk::enumerateInstanceExtensionProperties();
        std::cout << extensions.size() << " extensions supported" << std::endl;
        for (auto& ext : extensions )
        {
            std::cout<<ext.extensionName<<" "<<ext.specVersion<<std::endl;
        }
        std::cout << "=======================" << std::endl;
        std::cout << "Instance Layers" << std::endl;
        std::cout << "=======================" << std::endl;
        //Layers
        auto layers = vk::enumerateInstanceLayerProperties();
        std::cout << layers.size() << " layers supported" << std::endl;
        for (auto& ext : layers )
        {
            std::cout<<ext.layerName<<" "<<ext.implementationVersion<<std::endl;
            std::cout<<ext.description<<std::endl;
        }

        std::cout << "=======================" << std::endl;
        std::cout << "Physical Devices" << std::endl;
        std::cout << "=======================" << std::endl;
        // Physical Devices
        auto physicalDevices = instance.enumeratePhysicalDevices();
        device = physicalDevices[0];

        queueIndex = 0;
        presentQueue = 0;
        unsigned int i = 0;
        for( auto& dev : physicalDevices)
        {        
            std::cout << dev.getProperties().deviceName << std::endl;            
            std::cout << vk::to_string(dev.getProperties().deviceType) << std::endl;
            std::cout << VK_VERSION_MAJOR(dev.getProperties().driverVersion)<<"."<<VK_VERSION_MINOR(dev.getProperties().driverVersion) << std::endl;
            std::cout << VK_VERSION_MAJOR(dev.getProperties().apiVersion)<<"."<<VK_VERSION_MINOR(dev.getProperties().apiVersion) <<"."<<VK_VERSION_PATCH(dev.getProperties().apiVersion)<< std::endl;    

            // Get Queues
            std::vector<vk::QueueFamilyProperties> familyProperties = dev.getQueueFamilyProperties();
            for (auto& prop : familyProperties)
            {
                // Check if it supports presentation            
                std::cout<<"Queue Found,  index " <<i<<std::endl;
                if (dev.getSurfaceSupportKHR(i, surface))
                {
                    presentQueue = i;
                    std::cout<< "Surface Support found " <<  std::endl;
                }
                if(prop.queueFlags & vk::QueueFlagBits::eGraphics) 
                {                
                    queueIndex = i;
                    std::cout<< "eGraphics | ";
                }
                if(prop.queueFlags & vk::QueueFlagBits::eCompute)            
                std::cout<< "eCompute | ";
                if(prop.queueFlags & vk::QueueFlagBits::eProtected)            
                std::cout<< "eProtected | ";
                if(prop.queueFlags & vk::QueueFlagBits::eTransfer)            
                std::cout<< "eTransfer | ";
                std::cout<<std::endl<<prop.queueCount<<std::endl;
                ++i;
            }
            auto devEXTs = dev.enumerateDeviceExtensionProperties();

            std::cout << "Extenstions" << std::endl;
            std::cout << "-----------------" << std::endl;
            for (const auto& ext : devEXTs)
            {
                std::string temp = ext.extensionName;
                if ( temp == VK_KHR_SWAPCHAIN_EXTENSION_NAME )
                {
                    std::cout<< "Swapchain Support found " <<  std::endl;
                }
                std::cout << ext.extensionName << " " << ext.specVersion << std::endl;
            }
           
        }
        assert(presentQueue ==  queueIndex && "Presentation Queue differs from Graphic Queues,  engine currently does not support device");
        float priority = 1.0f;
        vk::DeviceQueueCreateInfo dqci(vk::DeviceQueueCreateFlags(), queueIndex, 1, &priority);
        vk::DeviceCreateInfo dci(vk::DeviceCreateFlags(), 1,  &dqci, validationLayers.size(), validationLayers.data(), devEXTsNeeded.size(), devEXTsNeeded.data());

        gpu = device.createDevice(dci);  
        
        vk::CommandPoolCreateInfo cpci(vk::CommandPoolCreateFlags(), queueIndex);
        commandPool = gpu.createCommandPool(cpci);
        
        createSwapChain();
        
    }
    catch(std::exception& e)
    {
        std::cout<<e.what()<<std::endl;
    }
}


void drawFrame()
{  
    try
    {
        //graphicsQueue.waitIdle();     //30% slower vs fences method below due to pipelining via fences (next frame can be sent and begin render while waiting)
        gpu.waitForFences(cpuSyncFN[currentFrame], 1, std::numeric_limits<uint64_t>::max());    
        // Reset after potential resize(recreateSwapChain) operation to prevent stall
        gpu.resetFences(cpuSyncFN[currentFrame]);
        
        auto fbIndex = gpu.acquireNextImageKHR(swapChain, std::numeric_limits<uint64_t>::max(), imgAvlblSMPH[currentFrame], nullptr);
    
        vk::PipelineStageFlags waitAt[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        vk::SubmitInfo si(1, &imgAvlblSMPH[currentFrame], waitAt, 1, &commandBuffers[fbIndex.value],1, &renderFinishSMPH[currentFrame]);
        std::vector<vk::SubmitInfo> subs;
        subs.push_back(si);
        graphicsQueue.submit(si, cpuSyncFN[currentFrame]);
        
        vk::PresentInfoKHR pi(1, &renderFinishSMPH[currentFrame], 1, &swapChain, &fbIndex.value);
        auto presResult = graphicsQueue.presentKHR(pi);
        currentFrame = (currentFrame + 1) %  MAX_FRAMES_IN_FLIGHT;
    }
    catch (vk::SystemError& err)
    {
        if (err.code() ==  vk::Result::eErrorOutOfDateKHR)
        {    // Check for resize
            std::cerr << "Resized! " << err.what() <<  std::endl;            
            recreateSwapChain();            
            return;
        }
    }
}

void createSemaphores()
{
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        imgAvlblSMPH.push_back(gpu.createSemaphore(vk::SemaphoreCreateInfo()));
        renderFinishSMPH.push_back(gpu.createSemaphore(vk::SemaphoreCreateInfo()));
        cpuSyncFN.push_back( gpu.createFence(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled)) );
    }
    
}

int main(int argc, char **argv) {

    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan window", nullptr, nullptr);
    VulkanStuff(); 
    createRenderPass();
    createPipeline();
    createFrameBuffers();
    createVertexBuffer();
    createCommandBuffers();
    createSemaphores();

    glm::mat4 matrix;
    glm::vec4 vec;
    auto test = matrix * vec;
    
    glfwSetWindowTitle(window, "Test");
    while(!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        ++FPS;
        if (now - lastFrame > 1.0f)
        {
            lastFrame = now;
            std::string t = boost::lexical_cast<std::string>(FPS);
            glfwSetWindowTitle(window, t.c_str());
            FPS = 0;
        }
        glfwPollEvents();
        drawFrame();      
      }

    gpu.waitIdle();
    cleanUpSwapChain();
    gpu.destroy(VBO);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        gpu.destroy(imgAvlblSMPH[i]);
        gpu.destroy(renderFinishSMPH[i]);
        gpu.destroy(cpuSyncFN[i]);
    }
    gpu.destroy(commandPool);
    gpu.destroy();

    instance.destroySurfaceKHR(surface);
	if (enableValidationLayers)
	{
		instance.destroyDebugUtilsMessengerEXT(debugMessenger, nullptr, dldy);
	}
    instance.destroy();
    glfwDestroyWindow(window);

    glfwTerminate();

    return 0;
  }
