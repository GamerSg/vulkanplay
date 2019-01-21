#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>
#include <vulkan/vulkan.hpp>

void VulkanStuff()
{
    auto appInfo = vk::ApplicationInfo("VulkanPlay",VK_MAKE_VERSION(1, 0, 0),"SYPX Engine",VK_MAKE_VERSION(1, 0, 0),VK_API_VERSION_1_0 );
    auto instance = vk::createInstance(vk::InstanceCreateInfo( vk::InstanceCreateFlags(), &appInfo ));
    auto physicalDevices = instance.enumeratePhysicalDevices();
    vk::PhysicalDevice& device = physicalDevices[0];
    for( auto& dev : physicalDevices)
    {        
        std::cout << dev.getProperties().deviceName << std::endl;            
        std::cout << vk::to_string(dev.getProperties().deviceType) << std::endl;
        std::cout << VK_VERSION_MAJOR(dev.getProperties().driverVersion)<<"."<<VK_VERSION_MINOR(dev.getProperties().driverVersion) << std::endl;
        std::cout << VK_VERSION_MAJOR(dev.getProperties().apiVersion)<<"."<<VK_VERSION_MINOR(dev.getProperties().apiVersion) <<"."<<VK_VERSION_PATCH(dev.getProperties().apiVersion)<< std::endl;    
        
        std::vector<vk::QueueFamilyProperties> familyProperties = dev.getQueueFamilyProperties();
        for (auto& prop : familyProperties)
        {
            std::cout<<"Prop Found"<<std::endl;
            if(prop.queueFlags & vk::QueueFlagBits::eGraphics)            
                std::cout<< "eGraphics | ";
            if(prop.queueFlags & vk::QueueFlagBits::eCompute)            
                std::cout<< "eCompute | ";
            if(prop.queueFlags & vk::QueueFlagBits::eProtected)            
                std::cout<< "eProtected | ";
            if(prop.queueFlags & vk::QueueFlagBits::eTransfer)            
                std::cout<< "eTransfer | ";
            std::cout<<std::endl<<prop.queueCount<<std::endl;
        }
    }
}

int main(int argc, char **argv) {
    VulkanStuff();
    glfwInit();
    
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);
    
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::cout << extensionCount << " extensions supported" << std::endl;

    glm::mat4 matrix;
    glm::vec4 vec;
    auto test = matrix * vec;

    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();
    }

    glfwDestroyWindow(window);

    glfwTerminate();
  
    return 0;
}
