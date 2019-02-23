#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
// Force depth range 0-1 for Vulkan,  OpenGL is -1 to 1
#define GLM_FORCE_DEPTH_ZERO_TO_ONE  
#include <glm/glm.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <chrono>
#include <bitset>
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
vk::Image depthImg;
vk::ImageView depthImgView;
vk::DeviceMemory depthMem;

vk::Buffer VBO;
vk::DeviceMemory devMem;
vk::Buffer IndicesBuffer;
vk::DeviceMemory indMem;
vk::Image texture;
vk::DeviceMemory texMem;

vk::ImageView texImgView;
vk::Sampler texSampler;

// Multiple for each frame
std::vector<vk::Buffer> uniformBuffer;
std::vector<vk::DeviceMemory> uboMem;

vk::DescriptorPool descPool;
vk::DescriptorSetLayout descSetLayout;
std::vector<vk::DescriptorSet> descSets;

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
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 uv;    
    
    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex);     

        return bindingDescription;
    }
    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
    std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions = {};
    
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);
    
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
    attributeDescriptions[1].offset = offsetof(Vertex, color);
    
    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = vk::Format::eR32G32Sfloat;
    attributeDescriptions[2].offset = offsetof(Vertex, uv);

    return attributeDescriptions;
    }
};

struct UniformBufferObject {    
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;    
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}}, 
    {{0.5f, -0.5f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f}}, 
    
    {{-0.5f, -0.5f, -0.4f}, {1.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, -0.4f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.4f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}}, 
    {{0.5f, -0.5f, -0.4f}, {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f}} 
};

const std::vector<uint16_t> indices = {
   4, 5, 6, 4, 7, 5, 0, 1, 2, 0, 3, 1
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

//@{
//typeFilter -> What VBO needs
// properties -> What app needs
//@}
uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {

    vk::PhysicalDeviceMemoryProperties pdmp = device.getMemoryProperties();
    std::cout << "Mem type needed : " << typeFilter << " " << std::bitset<16>(typeFilter) << std::endl;
    std::cout << "Mem Types : " << pdmp.memoryTypeCount << " Mem Heaps: " << pdmp.memoryHeapCount << std::endl;
    
    for (uint32_t i = 0; i < pdmp.memoryTypeCount; i++) 
    {
        if ((typeFilter & (1 << i)) && (pdmp.memoryTypes[i].propertyFlags & properties) & properties) 
        {
            std::cout << "Found " << i << " " << std::bitset<16>(1 << i) << std::endl;
            return i;
        }
    }
    throw std::runtime_error("Unable to find memory type");
}

void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memProperties, vk::Buffer& buffer,  vk::DeviceMemory& deviceMem)
{
    vk::BufferCreateInfo bci(vk::BufferCreateFlags(), size, usage, vk::SharingMode::eExclusive);
    buffer = gpu.createBuffer(bci);
    // Find out real memory requirements from GPU
    vk::MemoryRequirements mreq = gpu.getBufferMemoryRequirements(buffer);
    // Prepare allocation
    vk::MemoryAllocateInfo mai(mreq.size);
    // Find and assign a memory type that VBO requires as well as has host(CPU) access
    mai.setMemoryTypeIndex( findMemoryType(mreq.memoryTypeBits, memProperties) );
    // Allocate it
    deviceMem = gpu.allocateMemory(mai);
    // Bind VBO to allocated memory
    gpu.bindBufferMemory(buffer, deviceMem, 0);      
  
}


vk::CommandBuffer beginSingleTimeCommands()
{
    vk::CommandBufferAllocateInfo cbai(commandPool, vk::CommandBufferLevel::ePrimary, 1);    
    std::vector<vk::CommandBuffer> transferCmdBuffers = gpu.allocateCommandBuffers(cbai);
    vk::CommandBuffer& transferCmdBuffer = transferCmdBuffers[0];
    
    vk::CommandBufferBeginInfo cbbi(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    transferCmdBuffer.begin(cbbi);
    return transferCmdBuffer;
}

void endSingleTimeCommands(vk::CommandBuffer buff)
{
    buff.end();
    vk::SubmitInfo si;
    si.setCommandBufferCount(1);
    si.setPCommandBuffers(&buff);
    std::array<vk::SubmitInfo, 1> siArr = {si};
    graphicsQueue.submit(siArr, nullptr);
    graphicsQueue.waitIdle();                               // BAD,  needs to be reworked
    gpu.freeCommandBuffers(commandPool, 1, &buff);    
}


void copyBuffer(vk::Buffer src,  vk::Buffer dst,  vk::DeviceSize size)
{
    vk::CommandBuffer cb = beginSingleTimeCommands();
    
    vk::BufferCopy bc(0, 0, size);    
    std::array<vk::BufferCopy, 1> bcRegions=  {bc};
    cb.copyBuffer(src, dst, bcRegions);
    
    endSingleTimeCommands(cb); 
}

void copyBufferToImage(vk::Buffer src,  vk::Image img,  uint32_t width,  uint32_t height)
{
    vk::CommandBuffer cb = beginSingleTimeCommands();
    vk::ImageSubresourceLayers isl(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
    vk::BufferImageCopy bic(0, 0, 0, isl, vk::Offset3D(0, 0, 0), vk::Extent3D(width,height, 1));
    std::array <vk::BufferImageCopy, 1> bicArr = {bic};
    cb.copyBufferToImage(src, img, vk::ImageLayout::eTransferDstOptimal, bic);
    endSingleTimeCommands(cb);
}


bool hasStencilComponent(vk::Format format) 
{
    return (format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint);
}

void transitionImageLayout(vk::Image img,  vk::Format fmt,  vk::ImageLayout oldLayout,  vk::ImageLayout newLayout)
{
    vk::PipelineStageFlags srcStage,  dstStage;
    vk::CommandBuffer cb = beginSingleTimeCommands();
    vk::ImageMemoryBarrier imb(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eMemoryRead, oldLayout, newLayout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img);
    if (newLayout ==  vk::ImageLayout::eDepthStencilAttachmentOptimal)
    {
        imb.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);
        if (hasStencilComponent(fmt))
        {
            imb.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
        }
    }
    else
    {
        imb.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    }
    
    if (oldLayout ==  vk::ImageLayout::eUndefined && newLayout ==  vk::ImageLayout::eTransferDstOptimal)
    {
        srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
        dstStage = vk::PipelineStageFlagBits::eTransfer;
        imb.setSrcAccessMask({});                           //Doesnt really matter what we set because srcStage is TOP of Pipe,  ie. nothing to wait for
        imb.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
    }
    else if (oldLayout ==  vk::ImageLayout::eTransferDstOptimal && newLayout ==  vk::ImageLayout::eShaderReadOnlyOptimal)
    {  // Ensure Transfer is over before Fragment Shader can begin
        srcStage = vk::PipelineStageFlagBits::eTransfer;
        dstStage = vk::PipelineStageFlagBits::eFragmentShader;
        imb.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);     
        imb.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    }
    else if (oldLayout ==  vk::ImageLayout::eUndefined && newLayout ==  vk::ImageLayout::eDepthStencilAttachmentOptimal)
    {                                                       // Ensure Depth buffer 
        srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
        dstStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
        imb.setSrcAccessMask({});     
        imb.setDstAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite);        
    }
    
    std::array <vk::ImageMemoryBarrier, 1> imbArr = {imb};
    cb.pipelineBarrier(srcStage, dstStage, vk::DependencyFlagBits::eByRegion, nullptr, nullptr, imbArr);
    
    
    endSingleTimeCommands(cb);
    
}

void createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling,  vk::ImageUsageFlags usage, vk::MemoryPropertyFlags props, vk::Image& img,  vk::DeviceMemory& devM)
{
    vk::ImageCreateInfo ici(vk::ImageCreateFlags(), vk::ImageType::e2D, format, vk::Extent3D(width, height, 1), 1, 1, vk::SampleCountFlagBits::e1, tiling, usage, vk::SharingMode::eExclusive );
    
    img = gpu.createImage(ici);
    
    vk::MemoryRequirements memReq;
    memReq = gpu.getImageMemoryRequirements(img);
    vk::MemoryAllocateInfo mai( memReq.size, findMemoryType(memReq.memoryTypeBits, props) );
    devM = gpu.allocateMemory(mai);
    gpu.bindImageMemory(img, devM, 0);
}

void createTextureImage()
{
    int texW, texH, texCH;
    stbi_uc* data = stbi_load("./textures/texture.png", &texW, &texH, &texCH, STBI_rgb_alpha);
    
    vk::DeviceSize imgSize = texW * texH * 4;               // 4 channels,  rgba
    
    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingMem;
    
    createBuffer(imgSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible |  vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingMem);
    void* dat = gpu.mapMemory(stagingMem, 0, imgSize);  
    memcpy(dat, data, imgSize);
    gpu.unmapMemory(stagingMem);
    stbi_image_free(data);
    
    createImage(texW, texH, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled |  vk::ImageUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, texture, texMem);
      
    transitionImageLayout(texture, vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    copyBufferToImage(stagingBuffer, texture, texW, texH);
    transitionImageLayout(texture, vk::Format::eR8G8B8A8Unorm, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    
    gpu.destroy(stagingBuffer);
    gpu.freeMemory(stagingMem);
    
}

void createTextureSampler()
{
    vk::SamplerCreateInfo sci(vk::SamplerCreateFlags(), vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, 0, 1, 16.0f, 0, vk::CompareOp::eNever, 0, 0);
    texSampler = gpu.createSampler(sci);
}

vk::ImageView createImageView(vk::Image img,  vk::Format format,  vk::ImageAspectFlags aspect)
{
    vk::ImageViewCreateInfo ivci;
    ivci.flags = vk::ImageViewCreateFlags();
    ivci.image = img;
    ivci.viewType = vk::ImageViewType::e2D;
    ivci.format = format;
    ivci.subresourceRange.aspectMask = aspect;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;
    return gpu.createImageView(ivci);
}

void createTextureImageView()
{
    texImgView = createImageView(texture, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
}

void createUniformBuffers()
{
    vk::DeviceSize size = sizeof(UniformBufferObject);
    uniformBuffer.resize(swapImgs.size());
    uboMem.resize(swapImgs.size());
    
    for (int i = 0; i < uniformBuffer.size(); ++i)
    {
        createBuffer(size, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible |  vk::MemoryPropertyFlagBits::eHostCoherent, uniformBuffer[i], uboMem[i]);
    }
}

void createVertexBuffer()
{
    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingMem;
    size_t size = sizeof(Vertex) * vertices.size();
    createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible |  vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingMem);
    createBuffer(size, vk::BufferUsageFlagBits::eVertexBuffer |  vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, VBO, devMem);
    
    // Copy triangle to staging
    void* vboMem = gpu.mapMemory(stagingMem, 0, size);
    memcpy(vboMem, vertices.data(), size);
    gpu.unmapMemory(stagingMem);
    
    copyBuffer(stagingBuffer, VBO, size);
    gpu.destroy(stagingBuffer);
    gpu.freeMemory(stagingMem);
}

void createIndexBuffer()
{
    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingMem;
    size_t size = sizeof(uint16_t) * indices.size();
    createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible |  vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingMem);
    createBuffer(size, vk::BufferUsageFlagBits::eIndexBuffer |  vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, IndicesBuffer, indMem);
    
    // Copy triangle to staging
    void* Mem = gpu.mapMemory(stagingMem, 0, size);
    memcpy(Mem, indices.data(), size);
    gpu.unmapMemory(stagingMem);
    
    copyBuffer(stagingBuffer, IndicesBuffer, size);
    gpu.destroy(stagingBuffer);
    gpu.freeMemory(stagingMem);
}

void createDescriptorSetLayout()
{
    // Stuff we are passing to the shader
    vk::DescriptorSetLayoutBinding dslb(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex);
    vk::DescriptorSetLayoutBinding dslb2(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment);
    
    std::array<vk::DescriptorSetLayoutBinding, 2> dslbArr = {dslb, dslb2};
    
    vk::DescriptorSetLayoutCreateInfo dslci;
    dslci.setBindingCount(dslbArr.size());
    dslci.setPBindings(dslbArr.data());
    descSetLayout = gpu.createDescriptorSetLayout(dslci);    
}

vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
    for (vk::Format format : candidates) {
        vk::FormatProperties props;        
        props = device.getFormatProperties(format);        

        if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}


vk::Format findDepthFormat()
{
    return findSupportedFormat( {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint}, vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

void createDepthResources()
{
    vk::Format depthFmt = findDepthFormat();
    createImage(actualExtent.width, actualExtent.height, depthFmt, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImg, depthMem);
    depthImgView = createImageView(depthImg, depthFmt, vk::ImageAspectFlagBits::eDepth);
    transitionImageLayout(depthImg, depthFmt, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
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
    visci.setVertexAttributeDescriptionCount(vertexInputAttrDesc.size());
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
    vk::PipelineDepthStencilStateCreateInfo pdssci(vk::PipelineDepthStencilStateCreateFlags(), 1, 1, vk::CompareOp::eLess, 0);
    // MSAA
    vk::PipelineMultisampleStateCreateInfo msci;
    // Blending
    vk::PipelineColorBlendAttachmentState cbas(1, vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR |  vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA );
    vk::PipelineColorBlendStateCreateInfo sbsci(vk::PipelineColorBlendStateCreateFlags(), 0, vk::LogicOp::eCopy, 1, &cbas);
    // For setting uniform/constant variables for shaders
    vk::PipelineLayoutCreateInfo plci;
    plci.setSetLayoutCount(1);
    plci.setPSetLayouts(&descSetLayout);   
    pipelineLayout = gpu.createPipelineLayout(plci);
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
    gpci.setPDepthStencilState(&pdssci);
    gpci.setLayout(pipelineLayout);
    gpci.setRenderPass(renderPass);
    gpci.setSubpass(0);
    graphicsPipeline = gpu.createGraphicsPipeline(nullptr, gpci);
    
    
    
    gpu.destroy(vsm);
    gpu.destroy(fsm);
   
}

void createRenderPass()
{
    // Color attch
    vk::AttachmentDescription ad(vk::AttachmentDescriptionFlags(), sf.format, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);
    // Depth attch
    // Store op is DontCare as depth contents are not important after rendering completion
    vk::AttachmentDescription ad2(vk::AttachmentDescriptionFlags(), findDepthFormat(), vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    vk::AttachmentReference ar(0, vk::ImageLayout::eColorAttachmentOptimal);
    vk::AttachmentReference ar2(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    vk::SubpassDescription sp(vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics);
    sp.setColorAttachmentCount(1);
    sp.setPColorAttachments(&ar);
    sp.setPDepthStencilAttachment(&ar2);
    
    std::array<vk::AttachmentDescription, 2> adArr = {ad, ad2};
    vk::RenderPassCreateInfo rpci(vk::RenderPassCreateFlags(), 2, adArr.data(), 1, &sp);
    
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
            swapImgViews[i], 
            depthImgView
        };
        vk::FramebufferCreateInfo fbci(vk::FramebufferCreateFlags(), renderPass, 2, attch, actualExtent.width, actualExtent.height, 1);
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
            swapImgViews[i] = createImageView(swapImgs[i], sf.format, vk::ImageAspectFlagBits::eColor);
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
    gpu.destroy(depthImg);
    gpu.destroy(depthImgView);
    gpu.freeMemory(depthMem);
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
        std::array<float, 4> bgColor = {0.2f, 0.2f, 0.2f, 1.0f};        
        vk::ClearValue cv(bgColor);
        vk::ClearValue depthCV(vk::ClearDepthStencilValue(1.0f, 0) );
        
        std::array<vk::ClearValue, 2> cvArr =  {cv, depthCV};
        
        vk::RenderPassBeginInfo rpbi(renderPass, frameBuffers[i], scissors, 2, cvArr.data());
        commandBuffers[i].beginRenderPass(rpbi, vk::SubpassContents::eInline);          // Begin Render Pass        
        //Bind our painstakingly created pipeline
        commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline); 
        std::array<vk::Buffer, 1> vbos = {VBO};
        std::array<vk::DeviceSize, 1> offs = {0};
        std::array<vk::DescriptorSet, 1> dsArr = {descSets[i]};
        
        commandBuffers[i].bindVertexBuffers(0, vbos, offs);
        commandBuffers[i].bindIndexBuffer(IndicesBuffer, 0, vk::IndexType::eUint16);
        commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, dsArr, nullptr);
        commandBuffers[i].drawIndexed(indices.size(), 1, 0, 0, 0);
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
    createDepthResources();
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
        // Enable Anisotropic filtering for texture undersampling
        vk::PhysicalDeviceFeatures pdf;
        pdf.samplerAnisotropy = 1;
        vk::DeviceCreateInfo dci(vk::DeviceCreateFlags(), 1,  &dqci, validationLayers.size(), validationLayers.data(), devEXTsNeeded.size(), devEXTsNeeded.data(), &pdf);

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

void createDescriptorPool()
{
    vk::DescriptorPoolSize dps(vk::DescriptorType::eUniformBuffer, swapImgs.size());
    vk::DescriptorPoolSize dps2(vk::DescriptorType::eCombinedImageSampler, swapImgs.size());
    
    std::array<vk::DescriptorPoolSize, 2> dpsArr = {dps, dps2};
    
    vk::DescriptorPoolCreateInfo dpci;
    dpci.setPoolSizeCount(dpsArr.size());
    dpci.setPPoolSizes(dpsArr.data());
    dpci.setMaxSets(swapImgs.size());
    
    descPool = gpu.createDescriptorPool(dpci);
    
}

void createDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(swapImgs.size(), descSetLayout);
    vk::DescriptorSetAllocateInfo dsai(descPool, swapImgs.size(), layouts.data());
    
    descSets.resize(swapImgs.size());
    descSets = gpu.allocateDescriptorSets(dsai);
    
    for (int i = 0; i < descSets.size(); ++i)
    {
            vk::DescriptorBufferInfo dbi(uniformBuffer[i], 0,  sizeof(UniformBufferObject));            
            vk::WriteDescriptorSet wds(descSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &dbi, nullptr);
            vk::DescriptorImageInfo dii(texSampler, texImgView, vk::ImageLayout::eShaderReadOnlyOptimal);
            vk::WriteDescriptorSet wds2(descSets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &dii, nullptr, nullptr);
            std::array<vk::WriteDescriptorSet, 2> writeDescArr=  {wds, wds2};
            gpu.updateDescriptorSets(writeDescArr, nullptr);
    }
}

void updateUniformData(uint32_t currImg)
{
        UniformBufferObject ubo;
        static auto start = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float gameTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - start).count();
        ubo.model = glm::rotate(glm::identity<glm::mat4>(), glm::radians(gameTime * 90.0f), glm::vec3(0, 0, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(0.0, -2.0f, -2.0f), glm::vec3(0, 0, 0), glm::vec3(0, 1.0f, 0.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(actualExtent.width/actualExtent.height), 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;
        
        void* data = gpu.mapMemory(uboMem[currImg], 0, sizeof(ubo));
        memcpy(data, &ubo, sizeof(ubo));
        gpu.unmapMemory(uboMem[currImg]);        
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
        
        updateUniformData(fbIndex.value);
    
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
    createDepthResources();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    createRenderPass();
    createDescriptorSetLayout();
    createPipeline();
    createFrameBuffers();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSemaphores();
        
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
    gpu.destroy(descSetLayout);
    gpu.destroy(descPool);
    for (int i = 0; i < uniformBuffer.size(); ++i)
    {
        gpu.destroy(uniformBuffer[i]);
        gpu.freeMemory(uboMem[i]);
    }
    gpu.destroy(VBO);
    gpu.destroy(IndicesBuffer);
    gpu.destroy(texImgView);
    gpu.destroy(texSampler);
    gpu.destroy(texture);
    gpu.freeMemory(devMem);
    gpu.freeMemory(indMem);
    gpu.freeMemory(texMem);
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
