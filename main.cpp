#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <fstream>


GLFWwindow* window;
VkSurfaceKHR surface;                                       // Surface used by window to render to
vk::Instance instance;
vk::DispatchLoaderDynamic dldy;                             //Dynamic Loader for extensions
VkDebugUtilsMessengerEXT debugMessenger;
vk::Device gpu;
vk::PresentModeKHR presMode;

vk::SwapchainKHR swapChain;
std::vector<vk::Image> swapImgs;
std::vector<vk::ImageView> swapImgViews;
vk::PipelineLayout pipelineLayout;

const int WIDTH = 800;
const int HEIGHT = 600;

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
  
void createPipeline()
{
    auto VS = loadFileToMem("/shaders/vert.spv");
    auto FS = loadFileToMem("/shaders/frag.spv");
    
    vk::ShaderModule vsm = createShaderModule(VS);
    vk::ShaderModule fsm = createShaderModule(FS);
    
    vk::PipelineShaderStageCreateInfo vsStage(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex, vsm, "main");
    vk::PipelineShaderStageCreateInfo fsStage(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eFragment, fsm, "main");
    
    vk::PipelineVertexInputStateCreateInfo visci(vk::PipelineVertexInputStateCreateFlags());
    vk::PipelineInputAssemblyStateCreateInfo iasci(vk::PipelineInputAssemblyStateCreateFlags(), vk::PrimitiveTopology::eTriangleList);
    vk::Viewport viewport(0, 0, WIDTH, HEIGHT,  0,  1.0f);
    vk::Rect2D scissors;
    scissors.offset.x = 0;
    scissors.offset.y = 0;
    scissors.extent.width = WIDTH;
    scissors.extent.height = HEIGHT;
    vk::PipelineViewportStateCreateInfo vsci(vk::PipelineViewportStateCreateFlags(), 1, &viewport, 1, &scissors);
    vk::PipelineRasterizationStateCreateInfo rsci(vk::PipelineRasterizationStateCreateFlags(), 0, 0, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack,vk::FrontFace::eClockwise, 0, 0, 0, 0, 1.0f);
    // MSAA
    vk::PipelineMultisampleStateCreateInfo msci;
    vk::PipelineColorBlendAttachmentState cbas(1, vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR |  vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA );
    vk::PipelineColorBlendStateCreateInfo sbsci(vk::PipelineColorBlendStateCreateFlags(), 0, vk::LogicOp::eCopy, 1, &cbas);
    //vk::PipelineLayoutCreateInfo lci(vk::PipelineLayoutCreateFlags());
    pipelineLayout = gpu.createPipelineLayout(vk::PipelineLayoutCreateInfo());
    
    
    
    gpu.destroy(vsm);
    gpu.destroy(fsm);
   
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
        vk::PhysicalDevice& device = physicalDevices[0];

        unsigned int queueIndex, presentQueue, i = 0;
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
        
        // Create Swapchain
         std::cout << "Surface props" << std::endl;
        std::cout << "-----------------" << std::endl;

        auto surfCap = device.getSurfaceCapabilitiesKHR(surface);
        auto surfFormats = device.getSurfaceFormatsKHR(surface);
        auto surfPresModes = device.getSurfacePresentModesKHR(surface);
        vk::SurfaceFormatKHR sf;

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
            if (md ==  vk::PresentModeKHR::eFifoRelaxed)
            {
                presMode = vk::PresentModeKHR::eFifoRelaxed;                
                break;
            }
            presMode = md;
            std::cout <<  static_cast<unsigned int>(md) << std::endl;
        }

        vk::SwapchainCreateInfoKHR sci(vk::SwapchainCreateFlagsKHR(), surface, surfCap.minImageCount + 1, sf.format, sf.colorSpace,surfCap.currentExtent, 1, vk::ImageUsageFlags(vk::ImageUsageFlagBits::eColorAttachment), vk::SharingMode::eExclusive, 0, nullptr, surfCap.currentTransform, vk::CompositeAlphaFlagBitsKHR::eOpaque, presMode, VK_TRUE, nullptr); 
        swapChain = gpu.createSwapchainKHR(sci);
        swapImgs = gpu.getSwapchainImagesKHR(swapChain);
        for (auto & i :  swapImgs)
        {
            std::cout << "Swap Img : " << i << std::endl;
        }
        gpu.getQueue(0, 0);
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
    catch(std::exception& e)
    {
        std::cout<<e.what()<<std::endl;
    }
}

void WindowStuff()
{

}

int main(int argc, char **argv) {

    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan window", nullptr, nullptr);
    VulkanStuff(); 


    glm::mat4 matrix;
    glm::vec4 vec;
    auto test = matrix * vec;

    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();
      }

    for (auto &iv :  swapImgViews)
    {
        gpu.destroy(iv);
    }    
    gpu.destroy(pipelineLayout);
    gpu.destroy(swapChain);
    gpu.destroy();

    instance.destroySurfaceKHR(surface);
    instance.destroyDebugUtilsMessengerEXT(debugMessenger, nullptr, dldy);
    instance.destroy();
    glfwDestroyWindow(window);

    glfwTerminate();

    return 0;
  }
