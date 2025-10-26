#include "../base.h"
#include "../containers.h"
#include "../shader_interop/dense_geometry_shaderinterop.h"
#include "../memory.h"
#include "../thread_context.h"
#include <atomic>
#include "vulkan.h"
#include "../nvapi.h"
#include "../../third_party/nvapi/nvapi.h"
#include "../camera.h"
#define VMA_IMPLEMENTATION
#include "../../third_party/vulkan/vk_mem_alloc.h"

#ifdef USE_DLSS
PFun_slGetFeatureFunction *slGetFeatureFunction_p;
PFun_slSetTagForFrame *slSetTagForFrame_p;
PFun_slGetNewFrameToken *slGetNewFrameToken_p;
PFun_slEvaluateFeature *slEvaluateFeature_p;
PFun_slSetConstants *slSetConstants_p;
PFun_slAllocateResources *slAllocateResources_p;
PFun_slFreeResources *slFreeResources_p;
sl::Result slGetFeatureFunction(sl::Feature feature, const char *functionName, void *&function)
{
    return slGetFeatureFunction_p(feature, functionName, function);
}
#endif

namespace rt
{
Vulkan *device;

VkBool32 DebugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                     VkDebugUtilsMessageTypeFlagsEXT messageType,
                                     const VkDebugUtilsMessengerCallbackDataEXT *callbackData,
                                     void *userData)
{
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        Print("[Vulkan Warning]: %s\n", callbackData->pMessage);
    }
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        Print("[Vulkan Error]: %s\n", callbackData->pMessage);
    }
    else if (strcmp(callbackData->pMessageIdName, "WARNING_DEBUG_PRINTF"))
    {
        Print("Debug: %s\n", callbackData->pMessage);
    }
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
    {
        Print("[Vulkan Info]: %s\n", callbackData->pMessage);
    }

    return VK_FALSE;
}

void SLDebugMessengerCallback(sl::LogType logType, const char *msg) { Print(msg); }

Vulkan::Vulkan(ValidationMode validationMode, GPUDevicePreference preference) : frameCount(0)
{
    arena           = ArenaAlloc();
    const i32 major = 0;
    const i32 minor = 0;
    const i32 patch = 1;

#ifdef USE_DLSS
    sl::Preferences slPreference;
    slPreference.showConsole        = true;
    slPreference.logLevel           = sl::LogLevel::eVerbose;
    slPreference.logMessageCallback = SLDebugMessengerCallback;
    slPreference.flags              = sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    slPreference.applicationId      = 231313132;
    // slPreference.engine             = sl::EngineType::eCustom;
    // slPreference.engineVersion      = "RT";
    // slPreference.projectId          = "real_time_moana";

    sl::Feature slFeatures[]       = {sl::kFeatureDLSS_RR};
    slPreference.featuresToLoad    = slFeatures;
    slPreference.numFeaturesToLoad = ArrayLength(slFeatures);
    slPreference.renderAPI         = sl::RenderAPI::eVulkan;

    HMODULE module =
        LoadLibraryA("../../src/third_party/streamline/bin/x64/development/sl.interposer.dll");
    Assert(module);

    // note: function pointer is cast through void function pointer to silence
    // cast-function-type warning on gcc8

    PFun_slInit *slInit_p = (PFun_slInit *)GetProcAddress(module, "slInit");
    sl::Result result     = slInit_p(slPreference, sl::kSDKVersion);
    Assert(result == sl::Result::eOk);
    PFN_vkGetInstanceProcAddr slvkGetInstanceProcAddr =
        (PFN_vkGetInstanceProcAddr)(void (*)(void))GetProcAddress(module,
                                                                  "vkGetInstanceProcAddr");
    PFN_vkCreateDevice sl_vkCreateDevice =
        (PFN_vkCreateDevice)GetProcAddress(module, "vkCreateDevice");

    PFun_slIsFeatureSupported *slIsFeatureSupported_p =
        (PFun_slIsFeatureSupported *)GetProcAddress(module, "slIsFeatureSupported");
    slGetFeatureFunction_p =
        (PFun_slGetFeatureFunction *)GetProcAddress(module, "slGetFeatureFunction");
    slSetTagForFrame_p = (PFun_slSetTagForFrame *)GetProcAddress(module, "slSetTagForFrame");
    slGetNewFrameToken_p =
        (PFun_slGetNewFrameToken *)GetProcAddress(module, "slGetNewFrameToken");
    slSetConstants_p = (PFun_slSetConstants *)GetProcAddress(module, "slSetConstants");
    slEvaluateFeature_p =
        (PFun_slEvaluateFeature *)GetProcAddress(module, "slEvaluateFeature");
    slAllocateResources_p =
        (PFun_slAllocateResources *)GetProcAddress(module, "slAllocateResources");
    slFreeResources_p = (PFun_slFreeResources *)GetProcAddress(module, "slFreeResources");

    volkInitializeCustom(slvkGetInstanceProcAddr);
#endif
    volkInitialize();

    u32 apiVersion = VK_API_VERSION_1_4;

    // Create the application
    VkApplicationInfo appInfo  = {};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "RT";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    appInfo.pEngineName        = "RT";
    appInfo.engineVersion      = VK_MAKE_API_VERSION(0, major, minor, patch);
    appInfo.apiVersion         = apiVersion;

    // Load available layers
    u32 layerCount = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&layerCount, 0));
    std::vector<VkLayerProperties> availableLayers(layerCount);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()));

    // Load extension info
    u32 extensionCount = 0;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(0, &extensionCount, 0));
    std::vector<VkExtensionProperties> extensionProperties(extensionCount);
    VK_CHECK(vkEnumerateInstanceExtensionProperties(0, &extensionCount,
                                                    extensionProperties.data()));

    std::vector<const char *> instanceExtensions;
    std::vector<const char *> instanceLayers;
    // Add extensions
    for (auto &availableExtension : extensionProperties)
    {
        if (strcmp(availableExtension.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0)
        {
            debugUtils = true;
            instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
    }
    instanceExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef _WIN32
    instanceExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
#error not supported
#endif

    // Add layers
    if (validationMode != ValidationMode::Disabled)
    {
        static const std::vector<const char *> validationPriorityList[] = {
            // Preferred
            {"VK_LAYER_NV_optimus", "VK_LAYER_KHRONOS_validation"},
            // "VK_LAYER_KHRONOS_synchronization2"},
            // Fallback
            {"VK_LAYER_LUNARG_standard_validation"},
            // Individual
            {
                "VK_LAYER_GOOGLE_threading",
                "VK_LAYER_LUNARG_parameter_validation",
                "VK_LAYER_LUNARG_object_tracker",
                "VK_LAYER_LUNARG_core_validation",
                "VK_LAYER_GOOGLE_unique_objects",
            },
            // Last resort
            {
                "VK_LAYER_LUNARG_core_validation",
            },
        };
        for (auto &validationLayers : validationPriorityList)
        {
            bool validated = true;
            for (auto &layer : validationLayers)
            {
                bool found = false;
                for (auto &availableLayer : availableLayers)
                {
                    if (strcmp(availableLayer.layerName, layer) == 0)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    validated = false;
                    break;
                }
            }

            if (validated)
            {
                for (auto &c : validationLayers)
                {
                    instanceLayers.push_back(c);
                }
                break;
            }
        }
    }

    // Create instance
    {
        Assert(volkGetInstanceVersion() >= apiVersion);
        VkInstanceCreateInfo instInfo    = {};
        instInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instInfo.pApplicationInfo        = &appInfo;
        instInfo.enabledLayerCount       = (u32)instanceLayers.size();
        instInfo.ppEnabledLayerNames     = instanceLayers.data();
        instInfo.enabledExtensionCount   = (u32)instanceExtensions.size();
        instInfo.ppEnabledExtensionNames = instanceExtensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugUtilsCreateInfo = {};
        VkValidationFeaturesEXT validationFeatures              = {
            VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};

        ScratchArena scratch;
        debugUtilsCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

        StaticArray<VkValidationFeatureEnableEXT> validationFeaturesEnable(scratch.temp.arena,
                                                                           2);
        if (validationMode != ValidationMode::Disabled)
        {

            validationFeaturesEnable.Push(
                VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);
            validationFeaturesEnable.Push(VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT);
            validationFeatures.enabledValidationFeatureCount =
                validationFeaturesEnable.Length();
            validationFeatures.pEnabledValidationFeatures = validationFeaturesEnable.data;

            instInfo.pNext = &validationFeatures;

            if (debugUtils)
            {
                debugUtilsCreateInfo.messageSeverity =
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;

                debugUtilsCreateInfo.messageType =
                    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

                debugUtilsCreateInfo.pfnUserCallback = DebugUtilsMessengerCallback;
                validationFeatures.pNext             = &debugUtilsCreateInfo;
            }
        }

        VK_CHECK(vkCreateInstance(&instInfo, 0, &instance));
        volkLoadInstanceOnly(instance);

        if (validationMode != ValidationMode::Disabled && debugUtils)
        {
            VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance, &debugUtilsCreateInfo, 0,
                                                    &debugMessenger));
        }
    }

    // Enumerate physical devices
    {
        u32 deviceCount = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, 0));
        Assert(deviceCount != 0);

        std::vector<VkPhysicalDevice> devices(deviceCount);
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()));

        std::vector<const char *> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };

        VkPhysicalDevice preferred = VK_NULL_HANDLE;
        VkPhysicalDevice fallback  = VK_NULL_HANDLE;

        for (auto &testDevice : devices)
        {
            VkPhysicalDeviceProperties2 props = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            vkGetPhysicalDeviceProperties2(testDevice, &props);
            if (props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;

            u32 queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties2(testDevice, &queueFamilyCount, 0);

            std::vector<VkQueueFamilyProperties2> queueFamilyProps;
            queueFamilyProps.resize(queueFamilyCount);
            for (u32 i = 0; i < queueFamilyCount; i++)
            {
                queueFamilyProps[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
            }

            vkGetPhysicalDeviceQueueFamilyProperties2(testDevice, &queueFamilyCount,
                                                      queueFamilyProps.data());

            u32 graphicsIndex = VK_QUEUE_FAMILY_IGNORED;
            for (u32 i = 0; i < queueFamilyCount; i++)
            {
                if (queueFamilyProps[i].queueFamilyProperties.queueFlags &
                    VK_QUEUE_GRAPHICS_BIT)
                {
                    graphicsIndex = i;
                    break;
                }
            }
            if (graphicsIndex == VK_QUEUE_FAMILY_IGNORED) continue;

#ifdef _WIN32
            if (!vkGetPhysicalDeviceWin32PresentationSupportKHR(testDevice, graphicsIndex))
                continue;
#endif
            if (props.properties.apiVersion < apiVersion) continue;

#ifdef USE_DLSS
            sl::AdapterInfo adapterInfo;
            adapterInfo.vkPhysicalDevice = testDevice;
            sl::Result res = slIsFeatureSupported_p(sl::kFeatureDLSS_RR, adapterInfo);

            if (res != sl::Result::eOk) continue;
#endif

            b32 suitable = props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            if (preference == GPUDevicePreference::Integrated)
            {
                suitable =
                    props.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
            }
            if (!preferred && suitable)
            {
                preferred = testDevice;
            }
            if (!fallback)
            {
                fallback = testDevice;
            }
        }

        physicalDevice = preferred ? preferred : fallback;
        if (!physicalDevice)
        {
            Print("Error: No GPU selected\n");
            Assert(0);
        }
        // Printf("Selected GPU: %s\n", deviceProperties.properties.deviceName);

        deviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features11.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        features12.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        features13.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        features14.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
        deviceFeatures.pNext = &features11;
        features11.pNext     = &features12;
        features12.pNext     = &features13;
        void **featuresChain = &features13.pNext;

        deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties11.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
        properties12.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
        properties13.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
        properties14.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES;
        deviceProperties.pNext = &properties11;
        properties11.pNext     = &properties12;
        properties12.pNext     = &properties13;
        void **propertiesChain = &properties13.pNext;

        u32 deviceExtCount = 0;
        VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, 0, &deviceExtCount, 0));
        std::vector<VkExtensionProperties> availableDevExt(deviceExtCount);
        VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, 0, &deviceExtCount,
                                                      availableDevExt.data()));

        auto checkAndAddExtension = [&](const char *extName, void *prop = 0, void *feat = 0) {
            for (auto &extension : availableDevExt)
            {
                if (strcmp(extension.extensionName, extName) == 0)
                {
                    if (prop)
                    {
                        *propertiesChain = prop;
                        propertiesChain  = reinterpret_cast<void **>(
                            &reinterpret_cast<VkBaseOutStructure *>(prop)->pNext);
                    }
                    if (feat)
                    {
                        *featuresChain = feat;
                        featuresChain  = reinterpret_cast<void **>(
                            &reinterpret_cast<VkBaseOutStructure *>(feat)->pNext);
                    }
                    deviceExtensions.push_back(extName);
                    return true;
                }
            }
            return false;
        };

        checkAndAddExtension(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME);

        meshShaderProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
        meshShaderFeatures   = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        if (checkAndAddExtension(VK_EXT_MESH_SHADER_EXTENSION_NAME, &meshShaderProperties,
                                 &meshShaderFeatures))
        {
            capabilities |= DeviceCapabilities_MeshShader;
        }
        variableShadingRateProperties = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR};
        variableShadingRateFeatures = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR};
        if (checkAndAddExtension(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME,
                                 &variableShadingRateProperties, &variableShadingRateFeatures))
        {
            capabilities |= DeviceCapabilities_VariableShading;
        }

        // Ray tracing extensions
        {
            accelStructProps = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
            accelStructFeats = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
            bool result = checkAndAddExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                               &accelStructProps, &accelStructFeats);
            ErrorExit(result,
                      "Machine doesn't support VK_KHR_acceleration_structure. Exiting\n");

            rtPipeProperties = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
            rtPipeFeatures = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
            result = checkAndAddExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                                          &rtPipeProperties, &rtPipeFeatures);
            ErrorExit(result,
                      "Machine doesn't support VK_KHR_acceleration_structure. Exiting\n");

            result = checkAndAddExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
            Assert(result);

            rayQueryFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
            result =
                checkAndAddExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, 0, &rayQueryFeatures);
            Assert(result);

            reorderPropertiesNV = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_NV};
            reorderFeaturesNV = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV};

            result = checkAndAddExtension(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME,
                                          &reorderPropertiesNV, &reorderFeaturesNV);

            Assert(result);

            result = checkAndAddExtension(VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME);
            Assert(result);

            clasPropertiesNV = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};
            clasFeaturesNV = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV};
            result = checkAndAddExtension(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                          &clasPropertiesNV, &clasFeaturesNV);
            Assert(result);

            ptlasPropertiesNV = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_PROPERTIES_NV};
            ptlasFeaturesNV = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV};
            result =
                checkAndAddExtension(VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                     &ptlasPropertiesNV, &ptlasFeaturesNV);
            Assert(result);

            result = checkAndAddExtension(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
            Assert(result);
        }

        *featuresChain   = 0;
        *propertiesChain = 0;
        vkGetPhysicalDeviceFeatures2(physicalDevice, &deviceFeatures);

        // Ensure core functionlity is supported
        Assert(reorderFeaturesNV.rayTracingInvocationReorder == VK_TRUE);
        Assert(deviceFeatures.features.multiDrawIndirect == VK_TRUE);
        Assert(deviceFeatures.features.pipelineStatisticsQuery == VK_TRUE);
        Assert(features13.dynamicRendering == VK_TRUE);
        Assert(features12.descriptorIndexing == VK_TRUE);
        Assert(features12.bufferDeviceAddress == VK_TRUE);

        if (capabilities & DeviceCapabilities_MeshShader)
        {
            Assert(meshShaderFeatures.meshShader == VK_TRUE);
            Assert(meshShaderFeatures.taskShader == VK_TRUE);
        }

        vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProperties);
        cTimestampPeriod = (f64)deviceProperties.properties.limits.timestampPeriod * 1e-9;

        u32 queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice, &queueFamilyCount, 0);
        queueFamilyProperties =
            StaticArray<VkQueueFamilyProperties2>(arena, queueFamilyCount, queueFamilyCount);
        for (u32 i = 0; i < queueFamilyCount; i++)
        {
            queueFamilyProperties[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        }
        vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice, &queueFamilyCount,
                                                  queueFamilyProperties.data);

        // Device exposes 1+ queue families, queue families have 1+ queues. Each family
        // supports a combination of the below:
        // 1. Graphics
        // 2. Compute
        // 3. Transfer
        // 4. Sparse Memory Management

        // Find queues in queue family
        bool foundCopy    = false;
        bool foundCompute = false;
        for (u32 i = 0; i < queueFamilyProperties.Length(); i++)
        {
            auto &queueFamily = queueFamilyProperties[i];
            if (queueFamily.queueFamilyProperties.queueCount > 0)
            {
                if (graphicsFamily == VK_QUEUE_FAMILY_IGNORED &&
                    queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                {
                    graphicsFamily = i;
                }
                if (!foundCopy &&
                    (queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                    (copyFamily == VK_QUEUE_FAMILY_IGNORED ||
                     (!(queueFamily.queueFamilyProperties.queueFlags &
                        VK_QUEUE_GRAPHICS_BIT) &&
                      !(queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_COMPUTE_BIT))))

                {
                    if (!(queueFamily.queueFamilyProperties.queueFlags &
                          VK_QUEUE_GRAPHICS_BIT) &&
                        !(queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_COMPUTE_BIT))
                    {
                        foundCopy = true;
                    }
                    copyFamily = i;
                }
                if (!foundCompute &&
                    (queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                    (computeFamily == VK_QUEUE_FAMILY_IGNORED ||
                     !(queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_GRAPHICS_BIT)))

                {
                    if (computeFamily == VK_QUEUE_FAMILY_IGNORED ||
                        !(queueFamily.queueFamilyProperties.queueFlags &
                          VK_QUEUE_GRAPHICS_BIT))
                    {
                        foundCompute = true;
                    }
                    computeFamily = i;
                }
            }
        }

        // Create the device queues
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        families          = StaticArray<u32>(arena, QueueType_Count);
        f32 queuePriority = 1.f;
        for (u32 i = 0; i < QueueType_Count; i++)
        {
            u32 queueFamily = 0;
            if (i == QueueType_Graphics)
            {
                queueFamily = graphicsFamily;
            }
            else if (i == QueueType_Compute)
            {
                if (graphicsFamily == computeFamily)
                {
                    continue;
                }
                queueFamily = computeFamily;
            }
            else if (i == QueueType_Copy)
            {
                if (graphicsFamily == copyFamily || computeFamily == copyFamily)
                {
                    continue;
                }
                queueFamily = copyFamily;
            }
            VkDeviceQueueCreateInfo queueCreateInfo = {};
            queueCreateInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount       = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);

            families.Push(queueFamily);
        }

        VkDeviceCreateInfo createInfo      = {};
        createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount    = (u32)queueCreateInfos.size();
        createInfo.pQueueCreateInfos       = queueCreateInfos.data();
        createInfo.pEnabledFeatures        = 0;
        createInfo.pNext                   = &deviceFeatures;
        createInfo.enabledExtensionCount   = (u32)deviceExtensions.size();
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

#ifdef USE_DLSS
        VK_CHECK(sl_vkCreateDevice(physicalDevice, &createInfo, 0, &device));
#endif
        VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, 0, &device));

        volkLoadDevice(device);
    }

    memoryProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memoryProperties);

    // Get the device queues
    vkGetDeviceQueue(device, graphicsFamily, 0, &queues[QueueType_Graphics].queue);
    vkGetDeviceQueue(device, computeFamily, 0, &queues[QueueType_Compute].queue);
    vkGetDeviceQueue(device, copyFamily, 0, &queues[QueueType_Copy].queue);

    SetName(queues[QueueType_Graphics].queue, "Graphics Queue");
    SetName(queues[QueueType_Compute].queue, "Compute Queue");
    SetName(queues[QueueType_Copy].queue, "Transfer Queue");

    for (int i = 0; i < QueueType_Count; i++)
    {
        queues[i].submissionID = 0;
        for (int frame = 0; frame < numActiveFrames; frame++)
        {
            VkFenceCreateInfo info = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VkResult result = vkCreateFence(device, &info, 0, &queues[i].submitFence[frame]);
            VK_CHECK(result);
            // Semaphore s                      = CreateSemaphore();
            // queues[i].submitSemaphore[frame] = s.semaphore;
        }
    }
    u32 numProcessors = OS_NumProcessors();
    commandPools      = StaticArray<ThreadPool>(arena, numProcessors, numProcessors);

    // TODO: unified memory access architectures
    memProperties       = {};
    memProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memProperties);

    VmaAllocatorCreateInfo allocCreateInfo = {};
    allocCreateInfo.physicalDevice         = physicalDevice;
    allocCreateInfo.device                 = device;
    allocCreateInfo.instance               = instance;
    allocCreateInfo.vulkanApiVersion       = apiVersion;
    allocCreateInfo.flags                  = VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT |
                            VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

#ifdef WIN32
    allocCreateInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_EXTERNAL_MEMORY_WIN32_BIT;
#endif

    VmaVulkanFunctions vulkanFunctions    = {};
    vulkanFunctions.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    allocCreateInfo.pVulkanFunctions      = &vulkanFunctions;

    VK_CHECK(vmaCreateAllocator(&allocCreateInfo, &allocator));

    // Set up dynamic pso
    dynamicStates = {
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_VIEWPORT,
    };

    dynamicStateInfo                   = {};
    dynamicStateInfo.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.dynamicStateCount = (u32)dynamicStates.size();
    dynamicStateInfo.pDynamicStates    = dynamicStates.data();

    // Init descriptor pool
    {
        VkDescriptorPoolSize poolSizes[2];

        u32 count = 0;
        // Uniform buffers
        poolSizes[count].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[count].descriptorCount = cPoolSize;
        count++;

        // Combined samplers
        poolSizes[count].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[count].descriptorCount = cPoolSize;

        VkDescriptorPoolCreateInfo createInfo = {};
        createInfo.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        createInfo.poolSizeCount              = count;
        createInfo.pPoolSizes                 = poolSizes;
        createInfo.maxSets                    = cPoolSize;

        VK_CHECK(vkCreateDescriptorPool(device, &createInfo, 0, &bindlessPool));
    }

    // Bindless descriptor pools
    {
        DescriptorType types[] = {
            DescriptorType::SampledImage,
            DescriptorType::StorageBuffer,
            DescriptorType::StorageImage,
        };
        for (int type = 0; type < ArrayLength(types); type++)
        {
            VkDescriptorType descriptorType = ConvertDescriptorType(types[type]);

            BindlessDescriptorPool &bindlessDescriptorPool = bindlessDescriptorPools[type];
            VkDescriptorPoolSize poolSize                  = {};
            poolSize.type                                  = descriptorType;
            if (types[type] ==
                DescriptorType::StorageBuffer) // || type ==
                                               // DescriptorType_StorageTexelBuffer)
            {
                poolSize.descriptorCount =
                    Min(10000u,
                        deviceProperties.properties.limits.maxDescriptorSetStorageBuffers / 4);
            }
            else if (types[type] == DescriptorType::SampledImage)
            {
                poolSize.descriptorCount =
                    deviceProperties.properties.limits.maxDescriptorSetSampledImages / 2;
            }
            else if (types[type] == DescriptorType::StorageImage)
            {
                poolSize.descriptorCount =
                    Min(10000u,
                        deviceProperties.properties.limits.maxDescriptorSetStorageImages / 4);
            }
            // else if (type == DescriptorType_UniformTexel)
            // {
            //     poolSize.descriptorCount =
            //         Min(10000u,
            //             deviceProperties.properties.limits.maxDescriptorSetUniformBuffers /
            //             4);
            // }
            bindlessDescriptorPool.descriptorCount = poolSize.descriptorCount;

            VkDescriptorPoolCreateInfo createInfo = {
                VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            createInfo.poolSizeCount = 1;
            createInfo.pPoolSizes    = &poolSize;
            createInfo.maxSets       = 1;
            createInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
            VK_CHECK(
                vkCreateDescriptorPool(device, &createInfo, 0, &bindlessDescriptorPool.pool));

            VkDescriptorSetLayoutBinding binding = {};
            binding.binding                      = 0;
            binding.pImmutableSamplers           = 0;
            binding.stageFlags                   = VK_SHADER_STAGE_ALL;
            binding.descriptorType               = descriptorType;
            binding.descriptorCount              = bindlessDescriptorPool.descriptorCount;

            // These flags enable bindless:
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorBindingFlagBits.html
            VkDescriptorBindingFlags bindingFlags =
                VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
            VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCreate = {};
            bindingFlagsCreate.sType =
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
            bindingFlagsCreate.bindingCount  = 1;
            bindingFlagsCreate.pBindingFlags = &bindingFlags;

            VkDescriptorSetLayoutCreateInfo createSetLayout = {};
            createSetLayout.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            createSetLayout.bindingCount = 1;
            createSetLayout.pBindings    = &binding;
            createSetLayout.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
            createSetLayout.pNext = &bindingFlagsCreate;

            VK_CHECK(vkCreateDescriptorSetLayout(device, &createSetLayout, 0,
                                                 &bindlessDescriptorPool.layout));

            VkDescriptorSetAllocateInfo allocInfo = {};
            allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool     = bindlessDescriptorPool.pool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts        = &bindlessDescriptorPool.layout;
            VK_CHECK(
                vkAllocateDescriptorSets(device, &allocInfo, &bindlessDescriptorPool.set));

            for (u32 i = 0; i < poolSize.descriptorCount; i++)
            {
                bindlessDescriptorPool.freeList.push_back(poolSize.descriptorCount - i - 1);
            }
            bindlessDescriptorSets.push_back(bindlessDescriptorPool.set);
            bindlessDescriptorSetLayouts.push_back(bindlessDescriptorPool.layout);

            // Set debug names
            // TempArena temp = ScratchStart(0, 0);
            string typeName;
            switch (types[type])
            {
                case DescriptorType::SampledImage: typeName = "Sampled Image"; break;
                case DescriptorType::StorageBuffer: typeName = "Storage Buffer"; break;
                case DescriptorType::StorageImage: typeName = "Storage Image"; break;
                // case DescriptorType_UniformTexel: typeName = "Uniform Texel Buffer"; break;
                // case DescriptorType_StorageTexelBuffer:
                //     typeName = "Storage Texel Buffer";
                //     break;
                default: Assert(0);
            }
            string name = {};
            // PushStr8F(temp.arena, "Bindless Descriptor Set Layout: %S", typeName);
            SetName(bindlessDescriptorPool.layout, (const char *)name.str);

            // name = PushStr8F(temp.arena, "Bindless Descriptor Set: %S", typeName);
            SetName(bindlessDescriptorPool.set, (const char *)name.str);
            // ScratchEnd(temp);
        }
    }

    // Init frame allocators
#if 0
    {
        GPUBufferDesc desc;
        desc.usage         = MemoryUsage::CPU_TO_GPU;
        desc.size          = megabytes(32);
        desc.resourceUsage = ResourceUsage_TransferSrc;
        for (u32 i = 0; i < cNumBuffers; i++)
        {
            CreateBuffer(&frameAllocator[i].buffer, desc, 0);
            frameAllocator[i].alignment = 16;
        }
    }

    // Initialize ring buffer
    {
        u32 ringBufferSize = megabytes(128);
        GPUBufferDesc desc;
        desc.usage         = MemoryUsage::CPU_TO_GPU;
        desc.size          = ringBufferSize;
        desc.resourceUsage = ResourceUsage_TransferSrc;

        for (u32 i = 0; i < ArrayLength(stagingRingAllocators); i++)
        {
            RingAllocator &stagingRingAllocator = stagingRingAllocators[i];
            CreateBuffer(&stagingRingAllocator.transferRingBuffer, desc, 0);
            SetName(&stagingRingAllocator.transferRingBuffer, "Transfer Staging Buffer");

            stagingRingAllocator.ringBufferSize = ringBufferSize;
            stagingRingAllocator.writePos = stagingRingAllocator.readPos = 0;
            stagingRingAllocator.allocationReadPos                       = 0;
            stagingRingAllocator.allocationWritePos                      = 0;
            stagingRingAllocator.alignment                               = 16;
            stagingRingAllocator.lock.Init();
        }
    }
#endif

    // Default samplers
    {
        // Null sampler
        VkSamplerCreateInfo samplerCreate = {};
        samplerCreate.sType               = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

        // VK_CHECK(vkCreateSampler(device, &samplerCreate, 0, &nullSampler));

        samplerCreate.anisotropyEnable        = VK_FALSE;
        samplerCreate.maxAnisotropy           = 0;
        samplerCreate.minLod                  = 0;
        samplerCreate.maxLod                  = FLT_MAX;
        samplerCreate.mipLodBias              = 0;
        samplerCreate.unnormalizedCoordinates = VK_FALSE;
        samplerCreate.compareEnable           = VK_FALSE;
        samplerCreate.compareOp               = VK_COMPARE_OP_NEVER;
#if 0

        samplerCreate.minFilter    = VK_FILTER_LINEAR;
        samplerCreate.magFilter    = VK_FILTER_LINEAR;
        samplerCreate.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerCreate.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerCreate.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerCreate.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

        // sampler linear wrap
        immutableSamplers.emplace_back();
        VK_CHECK(vkCreateSampler(device, &samplerCreate, 0, &immutableSamplers.back()));

        // samler nearest wrap
        samplerCreate.minFilter  = VK_FILTER_NEAREST;
        samplerCreate.magFilter  = VK_FILTER_NEAREST;
        samplerCreate.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        immutableSamplers.emplace_back();
        VK_CHECK(vkCreateSampler(device, &samplerCreate, 0, &immutableSamplers.back()));
#endif

        // sampler linear clamp
        samplerCreate.minFilter    = VK_FILTER_LINEAR;
        samplerCreate.magFilter    = VK_FILTER_LINEAR;
        samplerCreate.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerCreate.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreate.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreate.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        immutableSamplers.emplace_back();
        VK_CHECK(vkCreateSampler(device, &samplerCreate, 0, &immutableSamplers.back()));

        // sampler nearest clamp
        samplerCreate.minFilter  = VK_FILTER_NEAREST;
        samplerCreate.magFilter  = VK_FILTER_NEAREST;
        samplerCreate.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        immutableSamplers.emplace_back();
        VK_CHECK(vkCreateSampler(device, &samplerCreate, 0, &immutableSamplers.back()));

#if 0
        // sampler nearest compare
        samplerCreate.compareEnable = VK_TRUE;
        samplerCreate.compareOp     = VK_COMPARE_OP_GREATER_OR_EQUAL;
        immutableSamplers.emplace_back();
        VK_CHECK(vkCreateSampler(device, &samplerCreate, 0, &immutableSamplers.back()));
#endif
    }

    // Default views
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType             = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType         = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width      = 1;
        imageInfo.extent.height     = 1;
        imageInfo.extent.depth      = 1;
        imageInfo.mipLevels         = 1;
        imageInfo.arrayLayers       = 1;
        imageInfo.format            = VK_FORMAT_R8G8B8A8_UNORM;
        imageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

        VmaAllocationCreateInfo allocInfo = {};
        allocInfo.usage                   = VMA_MEMORY_USAGE_GPU_ONLY;
        VK_CHECK(vmaCreateImage(allocator, &imageInfo, &allocInfo, &nullImage2D,
                                &nullImage2DAllocation, 0));

        VkImageViewCreateInfo createInfo           = {};
        createInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount     = 1;
        createInfo.subresourceRange.baseMipLevel   = 0;
        createInfo.subresourceRange.levelCount     = 1;
        createInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.format                          = VK_FORMAT_R8G8B8A8_UNORM;

        createInfo.image    = nullImage2D;
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;

        VK_CHECK(vkCreateImageView(device, &createInfo, 0, &nullImageView2D));

        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        VK_CHECK(vkCreateImageView(device, &createInfo, 0, &nullImageView2DArray));
    }

    // Null buffer
    {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size               = 4;
        bufferInfo.usage =
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo = {};
        allocInfo.preferredFlags          = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        VK_CHECK(vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &nullBuffer,
                                 &nullBufferAllocation, 0));
    }
}

Swapchain Vulkan::CreateSwapchain(OS_Handle window, VkFormat format, u32 width, u32 height)
{
    Swapchain swapchain     = {};
    swapchain.extent.width  = width;
    swapchain.extent.height = height;
    swapchain.format        = format;
#if _WIN32
    VkWin32SurfaceCreateInfoKHR win32SurfaceCreateInfo = {};
    win32SurfaceCreateInfo.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    win32SurfaceCreateInfo.hwnd      = (HWND)window.handle;
    win32SurfaceCreateInfo.hinstance = GetModuleHandleW(0);

    VK_CHECK(
        vkCreateWin32SurfaceKHR(instance, &win32SurfaceCreateInfo, 0, &swapchain.surface));
#else
#error not supported
#endif

    // Check whether physical device has a queue family that supports presenting to the surface
    u32 presentFamily = VK_QUEUE_FAMILY_IGNORED;
    for (u32 familyIndex = 0; familyIndex < queueFamilyProperties.Length(); familyIndex++)
    {
        VkBool32 supported = false;
        Assert(vkGetPhysicalDeviceSurfaceSupportKHR);
        VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, familyIndex,
                                                      swapchain.surface, &supported));

        if (queueFamilyProperties[familyIndex].queueFamilyProperties.queueCount > 0 &&
            supported)
        {
            presentFamily = familyIndex;
            break;
        }
    }
    if (presentFamily == VK_QUEUE_FAMILY_IGNORED)
    {
        return {};
    }

    CreateSwapchain(&swapchain);

    return swapchain;
}

b32 Vulkan::CreateSwapchain(Swapchain *swapchain)
{
    Assert(swapchain);

    u32 formatCount = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, swapchain->surface,
                                                  &formatCount, 0));
    std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, swapchain->surface,
                                                  &formatCount, surfaceFormats.data()));

    VkSurfaceCapabilitiesKHR surfaceCapabilities;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, swapchain->surface,
                                                       &surfaceCapabilities));

    u32 presentCount = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, swapchain->surface,
                                                       &presentCount, 0));
    std::vector<VkPresentModeKHR> surfacePresentModes;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(
        physicalDevice, swapchain->surface, &presentCount, surfacePresentModes.data()));

    VkSurfaceFormatKHR surfaceFormat = {};
    {
        surfaceFormat.format     = swapchain->format;
        surfaceFormat.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

        VkFormat requestedFormat = swapchain->format;

        b32 valid = false;
        for (auto &checkedFormat : surfaceFormats)
        {
            if (requestedFormat == checkedFormat.format)
            {
                surfaceFormat = checkedFormat;
                valid         = true;
                break;
            }
        }
        if (!valid)
        {
            swapchain->format = VK_FORMAT_B8G8R8A8_UNORM;
        }
    }

    // Pick the extent (size)
    {
        if (surfaceCapabilities.currentExtent.width != 0xFFFFFFFF &&
            surfaceCapabilities.currentExtent.height != 0xFFFFFFFF)
        {
            swapchain->extent = surfaceCapabilities.currentExtent;
        }
        else
        {
            swapchain->extent.width =
                Clamp(swapchain->extent.width, surfaceCapabilities.minImageExtent.width,
                      surfaceCapabilities.maxImageExtent.width);
            swapchain->extent.height =
                Clamp(swapchain->extent.height, surfaceCapabilities.minImageExtent.height,
                      surfaceCapabilities.maxImageExtent.height);
        }
    }
    u32 imageCount = Max(2u, surfaceCapabilities.minImageCount);
    if (surfaceCapabilities.maxImageCount > 0 &&
        imageCount > surfaceCapabilities.maxImageCount)
    {
        imageCount = surfaceCapabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapchainCreateInfo = {};
    {
        swapchainCreateInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        swapchainCreateInfo.surface          = swapchain->surface;
        swapchainCreateInfo.minImageCount    = imageCount;
        swapchainCreateInfo.imageFormat      = surfaceFormat.format;
        swapchainCreateInfo.imageColorSpace  = surfaceFormat.colorSpace;
        swapchainCreateInfo.imageExtent      = swapchain->extent;
        swapchainCreateInfo.imageArrayLayers = 1;
        swapchainCreateInfo.imageUsage =
            VK_IMAGE_USAGE_TRANSFER_DST_BIT; // VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swapchainCreateInfo.preTransform     = surfaceCapabilities.currentTransform;
        swapchainCreateInfo.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        // Choose present mode. Mailbox allows old images in swapchain queue to be replaced if
        // the queue is full.
        swapchainCreateInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        for (auto &presentMode : surfacePresentModes)
        {
            if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                swapchainCreateInfo.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
                break;
            }
        }
        swapchainCreateInfo.clipped      = VK_TRUE;
        swapchainCreateInfo.oldSwapchain = swapchain->swapchain;

        VK_CHECK(vkCreateSwapchainKHR(device, &swapchainCreateInfo, 0, &swapchain->swapchain));

        // Clean up the old swap chain, if it exists
        if (swapchainCreateInfo.oldSwapchain != VK_NULL_HANDLE)
        {
            u32 currentBuffer = GetCurrentBuffer();
            MutexScope(&cleanupMutex)
            {
                cleanupSwapchains[currentBuffer].push_back(swapchainCreateInfo.oldSwapchain);
                for (u32 i = 0; i < (u32)swapchain->imageViews.size(); i++)
                {
                    cleanupImageViews[currentBuffer].push_back(swapchain->imageViews[i]);
                }
                for (u32 i = 0; i < (u32)swapchain->acquireSemaphores.size(); i++)
                {
                    cleanupSemaphores[currentBuffer].push_back(
                        swapchain->acquireSemaphores[i]);
                }
                swapchain->acquireSemaphores.clear();
            }
        }

        // Get swapchain images
        VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain->swapchain, &imageCount, 0));
        swapchain->images.resize(imageCount);
        VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain->swapchain, &imageCount,
                                         swapchain->images.data()));
        for (u32 i = 0; i < swapchain->images.size(); i++)
        {
            SetName((u64)swapchain->images[i], VK_OBJECT_TYPE_IMAGE, "Swapchain Image");
        }

        // Create swap chain image views (determine how images are accessed)
#if 0
        swapchain->imageViews.resize(imageCount);
        for (u32 i = 0; i < imageCount; i++)
        {
            VkImageViewCreateInfo createInfo           = {};
            createInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image                           = swapchain->images[i];
            createInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format                          = surfaceFormat.format;
            createInfo.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel   = 0;
            createInfo.subresourceRange.levelCount     = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount     = 1;

            // TODO: delete old image view
            VK_CHECK(vkCreateImageView(device, &createInfo, 0, &swapchain->imageViews[i]));
        }
#endif

        // Create swap chain semaphores
        {
            VkSemaphoreCreateInfo semaphoreInfo = {};
            semaphoreInfo.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

            if (swapchain->acquireSemaphores.empty())
            {
                u32 size = (u32)swapchain->images.size();
                swapchain->acquireSemaphores.resize(size);
                for (u32 i = 0; i < size; i++)
                {
                    VK_CHECK(vkCreateSemaphore(device, &semaphoreInfo, 0,
                                               &swapchain->acquireSemaphores[i]));
                }
            }
            if (swapchain->releaseSemaphores.empty())
            {
                u32 size = (u32)swapchain->images.size();
                swapchain->releaseSemaphores.resize(size);
                for (u32 i = 0; i < size; i++)
                {
                    VK_CHECK(vkCreateSemaphore(device, &semaphoreInfo, 0,
                                               &swapchain->releaseSemaphores[i]));
                }
            }
        }
    }
    return true;
}

ImageLimits Vulkan::GetImageLimits()
{
    ImageLimits limits;
    limits.max1DImageDim = deviceProperties.properties.limits.maxImageDimension1D;
    limits.max2DImageDim = deviceProperties.properties.limits.maxImageDimension2D;
    limits.maxNumLayers  = deviceProperties.properties.limits.maxImageArrayLayers;
    return limits;
}

Semaphore Vulkan::CreateSemaphore()
{
    VkSemaphoreTypeCreateInfo timelineInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timelineInfo.semaphoreType             = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphoreInfo    = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphoreInfo.pNext                    = &timelineInfo;

    Semaphore semaphore = {};
    vkCreateSemaphore(device, &semaphoreInfo, 0, &semaphore.semaphore);

    return semaphore;
}

void Vulkan::DestroySemaphore(Semaphore sem) { vkDestroySemaphore(device, sem.semaphore, 0); }

void Vulkan::AllocateCommandBuffers(ThreadPool &pool, QueueType type)
{
    auto *node = pool.buffers[type].AddNode(ThreadPool::commandBufferPoolSize);
    VkCommandBufferAllocateInfo bufferInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    bufferInfo.commandPool                 = pool.pool[type];
    bufferInfo.commandBufferCount          = ThreadPool::commandBufferPoolSize;
    bufferInfo.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    VkCommandBuffer buffers[ThreadPool::commandBufferPoolSize];

    VK_CHECK(vkAllocateCommandBuffers(device, &bufferInfo, buffers));
    for (int i = 0; i < ThreadPool::commandBufferPoolSize; i++)
    {
        node->values[i]               = CommandBuffer();
        node->values[i].buffer        = buffers[i];
        pool.freeList[type].AddBack() = &node->values[i];
    }
}

void Vulkan::CheckInitializedThreadPool(int threadIndex)
{
    ThreadPool &pool = GetThreadPool(threadIndex);
    if (pool.arena == 0)
    {
        pool.arena = ArenaAlloc();
        pool.pool  = StaticArray<VkCommandPool>(pool.arena, QueueType_Count, QueueType_Count);
        pool.buffers  = CommandBufferPool(pool.arena, QueueType_Count, QueueType_Count);
        pool.freeList = CommandBufferFreeList(pool.arena, QueueType_Count, QueueType_Count);

        VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags                   = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                         VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        int pools[QueueType_Count];
        pools[QueueType_Graphics] = graphicsFamily;
        pools[QueueType_Compute]  = computeFamily;
        pools[QueueType_Copy]     = copyFamily;

        for (int i = 0; i < QueueType_Count; i++)
        {
            Assert(pools[i] != VK_QUEUE_FAMILY_IGNORED);
            poolInfo.queueFamilyIndex = pools[i];
            VK_CHECK(vkCreateCommandPool(device, &poolInfo, 0, &pool.pool[i]));
            pool.buffers[i]  = CommandBufferList(pool.arena);
            pool.freeList[i] = ChunkedLinkedList<CommandBuffer *>(pool.arena);
            AllocateCommandBuffers(pool, QueueType(i));
        }

        VkDescriptorPoolCreateInfo poolCreateInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        VkDescriptorPoolSize poolSize[7];
        poolSize[0].type            = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        poolSize[0].descriptorCount = 30;
        poolSize[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSize[1].descriptorCount = 30;
        poolSize[2].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize[2].descriptorCount = 60;
        poolSize[3].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize[3].descriptorCount = 550;
        poolSize[4].type            = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        poolSize[4].descriptorCount = 30;
        poolSize[5].type            = VK_DESCRIPTOR_TYPE_SAMPLER;
        poolSize[5].descriptorCount = 30;
        poolSize[6].type            = VK_DESCRIPTOR_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_NV;
        poolSize[6].descriptorCount = 1;

        poolCreateInfo.pPoolSizes    = poolSize;
        poolCreateInfo.poolSizeCount = ArrayLength(poolSize);
        poolCreateInfo.maxSets       = 500;

        vkCreateDescriptorPool(device, &poolCreateInfo, 0, &pool.descriptorPool[0]);
        vkCreateDescriptorPool(device, &poolCreateInfo, 0, &pool.descriptorPool[1]);
    }
}

CommandBuffer *Vulkan::BeginCommandBuffer(QueueType queue, string name)
{
    int threadIndex = GetThreadIndex();
    CheckInitializedThreadPool(threadIndex);
    ThreadPool &pool = GetThreadPool(threadIndex);

    if (pool.currentFrame < frameCount)
    {
        vkResetDescriptorPool(device, pool.descriptorPool[GetCurrentBuffer()], 0);
        pool.currentFrame = frameCount;
    }

    bool success = false;

    CommandBuffer *cmd = 0;
    for (auto *node = pool.freeList[queue].first; node != 0; node = node->next)
    {
        for (int i = 0; i < node->count; i++)
        {
            CommandBuffer *test = node->values[i];
            u64 value;
            if (test->semaphore != VK_NULL_HANDLE)
                vkGetSemaphoreCounterValue(device, test->semaphore, &value);
            if (test->semaphore == VK_NULL_HANDLE ||
                (value != ULLONG_MAX && value >= test->submissionID))
            {
                cmd = test;
                pool.freeList[queue].Pop(&node->values[i]);

                success = true;
                break;
            }
        }
        if (success) break;
    }

    if (!success)
    {
        AllocateCommandBuffers(pool, queue);
        pool.freeList[queue].Pop(&cmd);
    }

    Assert(cmd);

    cmd->semaphore    = VK_NULL_HANDLE;
    cmd->submissionID = 0;

    cmd->type = queue;
    VK_CHECK(vkResetCommandBuffer(cmd->buffer, 0));
    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    SetName(cmd->buffer, (const char *)name.str);
    VK_CHECK(vkBeginCommandBuffer(cmd->buffer, &beginInfo));

    return cmd;
}

void Vulkan::ResetDescriptorPool(u32 buffer)
{
    ThreadPool &pool = GetThreadPool(GetThreadIndex());
    vkResetDescriptorPool(device, pool.descriptorPool[buffer], 0);
}

void Vulkan::SubmitCommandBuffer(CommandBuffer *cmd, bool frame, bool parallel)
{
    Assert(cmd->semaphore != VK_NULL_HANDLE || frame);
    VK_CHECK(vkEndCommandBuffer(cmd->buffer));
    ScratchArena scratch;
    CommandQueue &queue = queues[cmd->type];
    VkSubmitInfo2 info  = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};

    u32 waitSize = static_cast<u32>(cmd->waitSemaphores.size());
    VkSemaphoreSubmitInfo *submitInfo =
        PushArray(scratch.temp.arena, VkSemaphoreSubmitInfo, waitSize);

    for (u32 i = 0; i < waitSize; i++)
    {
        submitInfo[i].sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        submitInfo[i].value     = cmd->waitSemaphores[i].signalValue;
        submitInfo[i].semaphore = cmd->waitSemaphores[i].semaphore;
        submitInfo[i].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }

    u32 signalSize = static_cast<u32>(cmd->signalSemaphores.size());
    VkSemaphoreSubmitInfo *signalSubmitInfo =
        PushArray(scratch.temp.arena, VkSemaphoreSubmitInfo, signalSize + frame);

    for (u32 i = 0; i < signalSize; i++)
    {
        signalSubmitInfo[i].sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        signalSubmitInfo[i].value     = cmd->signalSemaphores[i].signalValue;
        signalSubmitInfo[i].semaphore = cmd->signalSemaphores[i].semaphore;
        signalSubmitInfo[i].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
    // if (frame)
    // {
    //     signalSubmitInfo[signalSize].sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    //     u32 submissionID                       = queue.submissionID + 1;
    //     signalSubmitInfo[signalSize].value     = submissionID;
    //     signalSubmitInfo[signalSize].semaphore = queue.submitSemaphore[GetCurrentBuffer()];
    //     signalSubmitInfo[signalSize].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    // }

    VkCommandBufferSubmitInfo bufferSubmitInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    bufferSubmitInfo.commandBuffer = cmd->buffer;

    info.waitSemaphoreInfoCount   = waitSize;
    info.pWaitSemaphoreInfos      = submitInfo;
    info.signalSemaphoreInfoCount = signalSize; // + frame;
    info.pSignalSemaphoreInfos    = signalSubmitInfo;
    info.commandBufferInfoCount   = 1;
    info.pCommandBufferInfos      = &bufferSubmitInfo;

    if (parallel) BeginMutex(&queue.lock);
    vkQueueSubmit2(queue.queue, 1, &info,
                   frame ? queue.submitFence[GetCurrentBuffer()] : VK_NULL_HANDLE);
    if (parallel) EndMutex(&queue.lock);

    cmd->waitSemaphores.clear();
    cmd->signalSemaphores.clear();
    if (frame)
    {
        // cmd->semaphore    = queue.submitSemaphore[GetCurrentBuffer()];
        cmd->submissionID = queue.submissionID + 1;
    }

    int threadIndex                    = GetThreadIndex();
    ThreadPool &pool                   = GetThreadPool(threadIndex);
    pool.freeList[cmd->type].AddBack() = cmd;
}

VkImageMemoryBarrier2
Vulkan::ImageMemoryBarrier(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                           VkPipelineStageFlags2 srcStageMask,
                           VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 srcAccessMask,
                           VkAccessFlags2 dstAccessMask, VkImageAspectFlags aspectFlags,
                           QueueType fromQueue, QueueType toQueue, u32 baseMipLevel,
                           u32 levelCount, u32 baseArrayLayer, u32 layerCount)
{
    VkImageMemoryBarrier2 barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.image                 = image;
    barrier.oldLayout             = oldLayout;
    barrier.newLayout             = newLayout;
    barrier.srcStageMask          = srcStageMask;

    barrier.dstStageMask  = dstStageMask;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstAccessMask = dstAccessMask;

    barrier.subresourceRange.aspectMask     = aspectFlags;
    barrier.subresourceRange.baseMipLevel   = baseMipLevel;
    barrier.subresourceRange.levelCount     = levelCount;
    barrier.subresourceRange.baseArrayLayer = baseArrayLayer;
    barrier.subresourceRange.layerCount     = layerCount;
    barrier.srcQueueFamilyIndex =
        fromQueue == QueueType_Ignored ? VK_QUEUE_FAMILY_IGNORED : GetQueueFamily(fromQueue);
    barrier.dstQueueFamilyIndex =
        toQueue == QueueType_Ignored ? VK_QUEUE_FAMILY_IGNORED : GetQueueFamily(toQueue);
    return barrier;
}

void CommandBuffer::Barrier(VkPipelineStageFlags2 srcStage, VkPipelineStageFlags2 dstStage,
                            VkAccessFlags2 srcAccess, VkAccessFlags2 dstAccess)
{
    VkMemoryBarrier2 barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask     = srcStage;
    barrier.srcAccessMask    = srcAccess;
    barrier.dstStageMask     = dstStage;
    barrier.dstAccessMask    = dstAccess;

    memBarriers.push_back(barrier);
}

void CommandBuffer::Barrier(GPUImage *image, VkImageLayout oldLayout, VkImageLayout newLayout,
                            VkPipelineStageFlags2 srcStageMask,
                            VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 srcAccessMask,
                            VkAccessFlags2 dstAccessMask, QueueType fromQueue,
                            QueueType toQueue, u32 baseMipLevel, u32 levelCount,
                            u32 baseArrayLayer, u32 layerCount)
{
    VkImageMemoryBarrier2 barrier = device->ImageMemoryBarrier(
        image->image, oldLayout, newLayout, srcStageMask, dstStageMask, srcAccessMask,
        dstAccessMask, image->aspect, fromQueue, toQueue, baseMipLevel, levelCount,
        baseArrayLayer, layerCount);

    imageBarriers.push_back(barrier);

    image->lastLayout   = newLayout;
    image->lastPipeline = dstStageMask;
    image->lastAccess   = dstAccessMask;
}

void CommandBuffer::Barrier(GPUImage *image, VkImageLayout layout, VkPipelineStageFlags2 stage,
                            VkAccessFlags2 access)
{
    VkImageMemoryBarrier2 barrier = device->ImageMemoryBarrier(
        image->image, image->lastLayout, layout, image->lastPipeline, stage, image->lastAccess,
        access, image->aspect);

    imageBarriers.push_back(barrier);

    image->lastLayout   = layout;
    image->lastPipeline = stage;
    image->lastAccess   = access;
}

void CommandBuffer::Barrier(GPUBuffer *inBuffer, VkPipelineStageFlags2 stage,
                            VkAccessFlags2 access)
{
    VkBufferMemoryBarrier2 barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    barrier.srcStageMask           = inBuffer->lastStage;
    barrier.srcAccessMask          = inBuffer->lastAccess;
    barrier.dstStageMask           = stage;
    barrier.dstAccessMask          = access;
    barrier.srcQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer                 = inBuffer->buffer;
    barrier.offset                 = 0;
    barrier.size                   = VK_WHOLE_SIZE;

    bufferBarriers.push_back(barrier);

    inBuffer->lastStage  = stage;
    inBuffer->lastAccess = access;
}

void CommandBuffer::TransferWriteBarrier(GPUImage *image)
{
    Assert(image->lastLayout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    Barrier(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT);
}
void CommandBuffer::UAVBarrier(GPUImage *image)
{
    Barrier(image, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_WRITE_BIT);
}

void CommandBuffer::FlushBarriers()
{
    VkDependencyInfo dependencyInfo         = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependencyInfo.bufferMemoryBarrierCount = bufferBarriers.size();
    dependencyInfo.pBufferMemoryBarriers    = bufferBarriers.data();
    dependencyInfo.imageMemoryBarrierCount  = imageBarriers.size();
    dependencyInfo.pImageMemoryBarriers     = imageBarriers.data();
    dependencyInfo.memoryBarrierCount       = memBarriers.size();
    dependencyInfo.pMemoryBarriers          = memBarriers.data();
    vkCmdPipelineBarrier2(buffer, &dependencyInfo);

    memBarriers.clear();
    bufferBarriers.clear();
    imageBarriers.clear();
}

void CommandBuffer::PushConstants(PushConstant *pc, void *ptr, VkPipelineLayout layout)
{
    vkCmdPushConstants(buffer, layout, GetVulkanShaderStage(pc->stage), pc->offset, pc->size,
                       ptr);
}

void Vulkan::CopyFrameBuffer(Swapchain *swapchain, CommandBuffer *cmd, GPUBuffer *buffer)
{
    for (;;)
    {
        swapchain->acquireSemaphoreIndex =
            (swapchain->acquireSemaphoreIndex + 1) % (swapchain->acquireSemaphores.size());

        VkResult res = vkAcquireNextImageKHR(
            device, swapchain->swapchain, UINT64_MAX,
            swapchain->acquireSemaphores[swapchain->acquireSemaphoreIndex], VK_NULL_HANDLE,
            &swapchain->imageIndex);

        if (res != VK_SUCCESS)
        {
            if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
            {
                if (CreateSwapchain(swapchain))
                {
                    continue;
                }
                else
                {
                    ErrorExit(0, "Failed to create swapchain.\n");
                }
            }
        }
        else
        {
            break;
        }
    }

    VkImageMemoryBarrier2 barrier = {};
    // Set swapchain to transfer dst, image to transfer src
    {
        VkImageMemoryBarrier2 barriers[] = {
            ImageMemoryBarrier(swapchain->images[swapchain->imageIndex],
                               VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_NONE,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT),
        };

        VkDependencyInfo dependencyInfo        = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependencyInfo.imageMemoryBarrierCount = ArrayLength(barriers);
        dependencyInfo.pImageMemoryBarriers    = barriers;
        vkCmdPipelineBarrier2(cmd->buffer, &dependencyInfo);
    }

    // Copy framebuffer
    {
        BufferImageCopy copy = {};
        copy.extent          = Vec3u(swapchain->extent.width, swapchain->extent.height, 1);

        VkBufferImageCopy vkCopy;
        vkCopy.bufferOffset                    = copy.bufferOffset;
        vkCopy.bufferRowLength                 = copy.rowLength;
        vkCopy.bufferImageHeight               = copy.imageHeight;
        vkCopy.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        vkCopy.imageSubresource.mipLevel       = copy.mipLevel;
        vkCopy.imageSubresource.baseArrayLayer = copy.baseLayer;
        vkCopy.imageSubresource.layerCount     = copy.layerCount;
        vkCopy.imageOffset.x                   = copy.offset.x;
        vkCopy.imageOffset.y                   = copy.offset.y;
        vkCopy.imageOffset.z                   = copy.offset.z;
        vkCopy.imageExtent.width               = copy.extent.x;
        vkCopy.imageExtent.height              = copy.extent.y;
        vkCopy.imageExtent.depth               = copy.extent.z;
        vkCmdCopyBufferToImage(cmd->buffer, buffer->buffer,
                               swapchain->images[swapchain->imageIndex],
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &vkCopy);
    }

    // Set swapchain to present
    {
        VkImageMemoryBarrier2 barrier = ImageMemoryBarrier(
            swapchain->images[swapchain->imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_ACCESS_2_NONE, VK_IMAGE_ASPECT_COLOR_BIT);

        VkDependencyInfo dependencyInfo        = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependencyInfo.imageMemoryBarrierCount = 1;
        dependencyInfo.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd->buffer, &dependencyInfo);
    }

    cmd->Wait(Semaphore{swapchain->acquireSemaphores[swapchain->acquireSemaphoreIndex]});
    cmd->Signal(Semaphore{swapchain->releaseSemaphores[swapchain->imageIndex]});
    SubmitCommandBuffer(cmd, true);

    VkPresentInfoKHR presentInfo   = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = &swapchain->releaseSemaphores[swapchain->imageIndex];
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = &swapchain->swapchain;
    presentInfo.pImageIndices      = &swapchain->imageIndex;
    VkResult res = vkQueuePresentKHR(queues[QueueType_Graphics].queue, &presentInfo);

    if (res != VK_SUCCESS)
    {
        if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
        {
            b32 result = CreateSwapchain(swapchain);
            Assert(result);
        }
        else
        {
            Assert(0)
        }
    }
}

void Vulkan::CopyFrameBuffer(Swapchain *swapchain, CommandBuffer *cmd, GPUImage *image)
{
    for (;;)
    {
        swapchain->acquireSemaphoreIndex =
            (swapchain->acquireSemaphoreIndex + 1) % (swapchain->acquireSemaphores.size());

        VkResult res = vkAcquireNextImageKHR(
            device, swapchain->swapchain, UINT64_MAX,
            swapchain->acquireSemaphores[swapchain->acquireSemaphoreIndex], VK_NULL_HANDLE,
            &swapchain->imageIndex);

        if (res != VK_SUCCESS)
        {
            if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
            {
                if (CreateSwapchain(swapchain))
                {
                    continue;
                }
                else
                {
                    ErrorExit(0, "Failed to create swapchain.\n");
                }
            }
        }
        else
        {
            break;
        }
    }

    VkImageMemoryBarrier2 barrier = {};
    // Set swapchain to transfer dst, image to transfer src
    {
        VkImageMemoryBarrier2 barriers[] = {
            ImageMemoryBarrier(swapchain->images[swapchain->imageIndex],
                               VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_NONE,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT),
            ImageMemoryBarrier(image->image, image->lastLayout,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image->lastPipeline,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT, image->lastAccess,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT),
        };

        VkDependencyInfo dependencyInfo        = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependencyInfo.imageMemoryBarrierCount = ArrayLength(barriers);
        dependencyInfo.pImageMemoryBarriers    = barriers;
        vkCmdPipelineBarrier2(cmd->buffer, &dependencyInfo);

        image->lastLayout   = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        image->lastAccess   = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        image->lastPipeline = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    }

    // Copy framebuffer
    {
        VkImageBlit2 blitInfo                  = {VK_STRUCTURE_TYPE_IMAGE_BLIT_2};
        blitInfo.srcSubresource.baseArrayLayer = 0;
        blitInfo.srcSubresource.mipLevel       = 0;
        blitInfo.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        blitInfo.srcSubresource.layerCount     = 1;
        blitInfo.srcOffsets[1].x               = image->desc.width;
        blitInfo.srcOffsets[1].y               = image->desc.height;
        blitInfo.srcOffsets[1].z               = 1;
        blitInfo.dstSubresource.baseArrayLayer = 0;
        blitInfo.dstSubresource.mipLevel       = 0;
        blitInfo.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        blitInfo.dstSubresource.layerCount     = 1;
        blitInfo.dstOffsets[1].x               = (int)swapchain->extent.width;
        blitInfo.dstOffsets[1].y               = (int)swapchain->extent.height;
        blitInfo.dstOffsets[1].z               = 1;

        VkBlitImageInfo2 blit = {VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2};
        blit.srcImage         = image->image;
        blit.srcImageLayout   = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        blit.dstImage         = swapchain->images[swapchain->imageIndex];
        blit.dstImageLayout   = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        blit.regionCount      = 1;
        blit.pRegions         = &blitInfo;
        blit.filter           = VK_FILTER_LINEAR;

        vkCmdBlitImage2(cmd->buffer, &blit);
    }

    // Set swapchain to present
    {
        VkImageMemoryBarrier2 barrier = ImageMemoryBarrier(
            swapchain->images[swapchain->imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_ACCESS_2_NONE, VK_IMAGE_ASPECT_COLOR_BIT);

        VkDependencyInfo dependencyInfo        = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependencyInfo.imageMemoryBarrierCount = 1;
        dependencyInfo.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd->buffer, &dependencyInfo);
    }

    cmd->Wait(Semaphore{swapchain->acquireSemaphores[swapchain->acquireSemaphoreIndex]});
    cmd->Signal(Semaphore{swapchain->releaseSemaphores[swapchain->imageIndex]});
    SubmitCommandBuffer(cmd, true);

    VkPresentInfoKHR presentInfo   = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = &swapchain->releaseSemaphores[swapchain->imageIndex];
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = &swapchain->swapchain;
    presentInfo.pImageIndices      = &swapchain->imageIndex;
    VkResult res = vkQueuePresentKHR(queues[QueueType_Graphics].queue, &presentInfo);

    if (res != VK_SUCCESS)
    {
        if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
        {
            b32 result = CreateSwapchain(swapchain);
            Assert(result);
        }
        else
        {
            Assert(0)
        }
    }
}

ThreadPool &Vulkan::GetThreadPool(int threadIndex) { return commandPools[threadIndex]; }

GPUBuffer Vulkan::CreateBuffer(VkBufferUsageFlags flags, size_t totalSize, MemoryUsage usage)
{
    GPUBuffer buffer              = {};
    buffer.size                   = totalSize;
    VkBufferCreateInfo createInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    createInfo.size               = totalSize;
    createInfo.usage |=
        flags | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (families.Length() > 1)
    {
        createInfo.sharingMode           = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = families.Length();
        createInfo.pQueueFamilyIndices   = families.data;
    }
    else
    {
        createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.usage                   = VMA_MEMORY_USAGE_AUTO;
    switch (usage)
    {
        case MemoryUsage::CPU_TO_GPU:
        {
            createInfo.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }
        break;
        case MemoryUsage::GPU_TO_CPU:
        {
            allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
            createInfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        }
        break;
        default: break;
    }

    VkExternalMemoryHandleTypeFlags handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkExternalMemoryBufferCreateInfo exportInfo = {
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    exportInfo.handleTypes = handleType;

    if (usage == MemoryUsage::EXTERNAL)
    {
        if (!externalMemoryInitialized)
        {
            externalMemoryInitialized = true;
            InitializeExternalMemoryPool();
        }

        createInfo.pNext     = &exportInfo;
        allocCreateInfo.pool = externalPool;
    }

    VK_CHECK(vmaCreateBuffer(allocator, &createInfo, &allocCreateInfo, &buffer.buffer,
                             &buffer.allocation, 0));

    VkMemoryRequirements r;
    vkGetBufferMemoryRequirements(device, buffer.buffer, &r);

    buffer.req.size      = r.size;
    buffer.req.alignment = r.alignment;
    buffer.req.bits      = r.memoryTypeBits;

    if (usage != MemoryUsage::GPU_ONLY && usage != MemoryUsage::CPU_ONLY)
    {
        buffer.mappedPtr = buffer.allocation->GetMappedData();
    }
    return buffer;
}

VmaAllocation Vulkan::AllocateMemory(MemoryRequirements &r)
{
    VkMemoryRequirements req;
    req.size           = r.size;
    req.memoryTypeBits = r.bits;
    req.alignment      = r.alignment;

    VmaAllocationCreateInfo info = {};
    info.preferredFlags          = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (EnumHasAnyFlags(r.usage, MemoryUsage::EXTERNAL))
    {
        info.pool = externalPool;
    }

    VmaAllocation alloc;
    VK_CHECK(vmaAllocateMemory(allocator, &req, &info, &alloc, 0));
    return alloc;
}

void Vulkan::FreeMemory(VmaAllocation alloc) { vmaFreeMemory(allocator, alloc); }

void Vulkan::BindBufferMemory(VmaAllocation alloc, VkBuffer buffer, uint64_t offset)
{
    vmaBindBufferMemory2(allocator, alloc, offset, buffer, 0);
}
void Vulkan::BindImageMemory(VmaAllocation alloc, VkImage image, uint64_t offset)
{
    vmaBindImageMemory2(allocator, alloc, offset, image, 0);
}

bool Vulkan::IsAllocated(VmaAllocation alloc)
{
    bool result = alloc->GetMemory();
    return result;
}

GPUBuffer Vulkan::CreateAliasedBuffer(VkBufferUsageFlags flags, size_t totalSize,
                                      MemoryUsage memUsage)
{
    GPUBuffer buffer = {};
    buffer.size      = totalSize;

    VkBufferCreateInfo createInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    createInfo.size               = totalSize;
    createInfo.usage |=
        flags | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    if (families.Length() > 1)
    {
        createInfo.sharingMode           = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = families.Length();
        createInfo.pQueueFamilyIndices   = families.data;
    }
    else
    {
        createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VkExternalMemoryHandleTypeFlags handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkExternalMemoryBufferCreateInfo exportInfo = {
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    exportInfo.handleTypes = handleType;

    if (memUsage == MemoryUsage::EXTERNAL)
    {
        if (!externalMemoryInitialized)
        {
            externalMemoryInitialized = true;
            InitializeExternalMemoryPool();
        }

        createInfo.pNext = &exportInfo;
    }

    VK_CHECK(vkCreateBuffer(device, &createInfo, 0, &buffer.buffer));

    VkMemoryRequirements r;
    vkGetBufferMemoryRequirements(device, buffer.buffer, &r);

    buffer.req.size      = r.size;
    buffer.req.alignment = r.alignment;
    buffer.req.bits      = r.memoryTypeBits;

    if (EnumHasAnyFlags(memUsage, MemoryUsage::EXTERNAL))
    {
        buffer.req.bits = externalMemTypeIndex;
    }

    return buffer;
}

GPUImage Vulkan::CreateAliasedImage(ImageDesc desc)
{
    GPUImage image  = {};
    image.desc      = desc;
    image.aspect    = VK_IMAGE_ASPECT_COLOR_BIT;
    image.imageView = VK_NULL_HANDLE;

    VkImageCreateInfo imageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    switch (desc.imageType)
    {
        case ImageType::Type1D:
        case ImageType::Array1D:
        {
            imageInfo.imageType = VK_IMAGE_TYPE_1D;
        }
        break;
        case ImageType::Type2D:
        case ImageType::Array2D:
        {
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
        }
        break;
        case ImageType::Cubemap:
        {
            imageInfo.imageType = VK_IMAGE_TYPE_3D;
        }
        break;
        default: Assert(0);
    }
    imageInfo.extent.width  = desc.width;
    imageInfo.extent.height = desc.height;
    imageInfo.extent.depth  = desc.depth;
    imageInfo.mipLevels     = desc.numMips;
    imageInfo.arrayLayers   = desc.numLayers;
    imageInfo.format        = desc.format;
    imageInfo.tiling        = desc.tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage         = desc.imageUsage;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;

    ScratchArena scratch;
    u32 *imageQueueFamilies   = PushArrayNoZero(scratch.temp.arena, u32, families.Length());
    u32 imageQueueFamilyCount = 0;

    if (families.Length() > 1)
    {
        imageInfo.sharingMode           = VK_SHARING_MODE_CONCURRENT;
        imageInfo.queueFamilyIndexCount = families.Length();
        imageInfo.pQueueFamilyIndices   = families.data;
    }
    else
    {
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.flags   = VK_IMAGE_CREATE_ALIAS_BIT;

    VkImageFormatProperties imageFormatProperties;

    VkResult result = vkGetPhysicalDeviceImageFormatProperties(
        physicalDevice, imageInfo.format, imageInfo.imageType, imageInfo.tiling,
        imageInfo.usage, imageInfo.flags, &imageFormatProperties);

    ErrorExit(result == VK_SUCCESS, "Image format properties error: %u\n", result);

    VK_CHECK(vkCreateImage(device, &imageInfo, 0, &image.image));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, image.image, &req);

    MemoryRequirements memReq;
    memReq.alignment = req.alignment;
    memReq.bits      = req.memoryTypeBits;
    memReq.size      = req.size;

    image.req = memReq;

    // numSubresources = numSubresources == -1 ? desc.numMips : numSubresources;
    // if (numSubresources)
    //     image.subresources = StaticArray<GPUImage::Subresource>(arena, numSubresources);
    // CreateSubresource(&image);

    image.lastPipeline = VK_PIPELINE_STAGE_2_NONE;
    image.lastLayout   = VK_IMAGE_LAYOUT_UNDEFINED;
    image.lastAccess   = VK_ACCESS_2_NONE;
    return image;
}

GPUImage Vulkan::CreateImage(ImageDesc desc, int numSubresources, int ownedQueues)
{
    GPUImage image  = {};
    image.desc      = desc;
    image.aspect    = VK_IMAGE_ASPECT_COLOR_BIT;
    image.imageView = VK_NULL_HANDLE;

    VkImageCreateInfo imageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    switch (desc.imageType)
    {
        case ImageType::Type1D:
        case ImageType::Array1D:
        {
            imageInfo.imageType = VK_IMAGE_TYPE_1D;
        }
        break;
        case ImageType::Type2D:
        case ImageType::Array2D:
        {
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
        }
        break;
        case ImageType::Cubemap:
        {
            imageInfo.imageType = VK_IMAGE_TYPE_3D;
        }
        break;
        default: Assert(0);
    }
    imageInfo.extent.width  = desc.width;
    imageInfo.extent.height = desc.height;
    imageInfo.extent.depth  = desc.depth;
    imageInfo.mipLevels     = desc.numMips;
    imageInfo.arrayLayers   = desc.numLayers;
    imageInfo.format        = desc.format;
    imageInfo.tiling        = desc.tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage         = desc.imageUsage;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;

    ScratchArena scratch;
    u32 *imageQueueFamilies   = PushArrayNoZero(scratch.temp.arena, u32, families.Length());
    u32 imageQueueFamilyCount = 0;

    if (ownedQueues == -1 && families.Length() > 1)
    {
        imageInfo.sharingMode           = VK_SHARING_MODE_CONCURRENT;
        imageInfo.queueFamilyIndexCount = (u32)families.Length();
        imageInfo.pQueueFamilyIndices   = families.data;
    }
    else if (ownedQueues == -1 && families.Length() == 1)
    {
        imageInfo.sharingMode           = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.queueFamilyIndexCount = (u32)families.Length();
        imageInfo.pQueueFamilyIndices   = families.data;
    }
    else
    {
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        for (u32 queueType = (u32)QueueType_Graphics; queueType < (u32)QueueType_Count;
             queueType++)
        {
            if ((ownedQueues & (1u << (u32)queueType)) != 0)
            {
                u32 family = GetQueueFamily((QueueType)queueType);
                imageQueueFamilies[imageQueueFamilyCount++] = family;
            }
        }
        imageInfo.queueFamilyIndexCount = imageQueueFamilyCount;
        imageInfo.pQueueFamilyIndices   = imageQueueFamilies;
    }

    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.flags   = 0;

    VkImageFormatProperties imageFormatProperties;

    VkResult result = vkGetPhysicalDeviceImageFormatProperties(
        physicalDevice, imageInfo.format, imageInfo.imageType, imageInfo.tiling,
        imageInfo.usage, imageInfo.flags, &imageFormatProperties);

    ErrorExit(result == VK_SUCCESS, "Image format properties error: %u\n", result);

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage                   = VMA_MEMORY_USAGE_AUTO;

    if (desc.memUsage == MemoryUsage::CPU_TO_GPU)
    {
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;
        imageInfo.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }
    if (desc.memUsage == MemoryUsage::GPU_TO_CPU)
    {
        allocInfo.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        imageInfo.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }

    VK_CHECK(
        vmaCreateImage(allocator, &imageInfo, &allocInfo, &image.image, &image.allocation, 0));

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, image.image, &req);

    MemoryRequirements memReq;
    memReq.alignment = req.alignment;
    memReq.bits      = req.memoryTypeBits;
    memReq.size      = req.size;

    image.req = memReq;

    numSubresources = numSubresources == -1 ? desc.numMips : numSubresources;
    if (numSubresources)
        image.subresources = StaticArray<GPUImage::Subresource>(arena, numSubresources);
    CreateSubresource(&image);

    image.lastPipeline = VK_PIPELINE_STAGE_2_NONE;
    image.lastLayout   = VK_IMAGE_LAYOUT_UNDEFINED;
    image.lastAccess   = VK_ACCESS_2_NONE;

    return image;
}

int Vulkan::CreateSubresource(GPUImage *image, u32 baseMip, u32 numMips, u32 baseLayer,
                              u32 numLayers)
{
    VkImageViewCreateInfo createInfo = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    createInfo.format                = image->desc.format;
    createInfo.image                 = image->image;

    switch (image->desc.imageType)
    {
        case ImageType::Type1D:
        {
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_1D;
        }
        break;
        case ImageType::Array1D:
        {
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
        }
        break;
        case ImageType::Type2D:
        {
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        }
        break;
        case ImageType::Array2D:
        {
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        }
        break;
        case ImageType::Cubemap:
        {
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        }
        break;
    }

    createInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    createInfo.subresourceRange.baseMipLevel   = baseMip;
    createInfo.subresourceRange.levelCount     = numMips;
    createInfo.subresourceRange.baseArrayLayer = baseLayer;
    createInfo.subresourceRange.layerCount     = numLayers;

    int result = -1;
    if (baseLayer == 0 && numLayers == VK_REMAINING_ARRAY_LAYERS && baseMip == 0 &&
        numMips == VK_REMAINING_MIP_LEVELS)
    {
        Assert(image->imageView == VK_NULL_HANDLE);
        VK_CHECK(vkCreateImageView(device, &createInfo, 0, &image->imageView));
    }
    else
    {
        GPUImage::Subresource subresource = {};
        subresource.baseLayer             = baseLayer;
        subresource.numLayers             = numLayers;
        subresource.baseMip               = baseMip;
        subresource.numMips               = numMips;

        VK_CHECK(vkCreateImageView(device, &createInfo, 0, &subresource.imageView));
        image->subresources.Push(subresource);
        result = (int)(image->subresources.Length() - 1);
    }

    return result;
}

u32 Vulkan::GetImageSize(GPUImage *image) { return (u32)image->allocation->GetSize(); }

void Vulkan::DestroyBuffer(GPUBuffer *buffer)
{
    vmaDestroyBuffer(allocator, buffer->buffer, buffer->allocation);
}

void Vulkan::DestroyBufferHandle(GPUBuffer *buffer)
{
    vkDestroyBuffer(device, buffer->buffer, 0);
}

void Vulkan::DestroyBuffer(VkBuffer buffer) { vkDestroyBuffer(device, buffer, 0); }

void Vulkan::DestroyImage(GPUImage *image)
{
    if (image->imageView) vkDestroyImageView(device, image->imageView, 0);
    for (GPUImage::Subresource &s : image->subresources)
    {
        vkDestroyImageView(device, s.imageView, 0);
    }
    vmaDestroyImage(allocator, image->image, image->allocation);
}

void Vulkan::DestroyImageHandles(GPUImage *image)
{
    if (image->imageView) vkDestroyImageView(device, image->imageView, 0);
    for (GPUImage::Subresource &s : image->subresources)
    {
        vkDestroyImageView(device, s.imageView, 0);
    }
    vkDestroyImage(device, image->image, 0);
}

void Vulkan::DestroyAccelerationStructure(GPUAccelerationStructure *as)
{
    vkDestroyAccelerationStructureKHR(device, as->as, 0);
    DestroyBuffer(&as->buffer);
}

void Vulkan::DestroyAccelerationStructure(VkAccelerationStructureKHR as)
{
    vkDestroyAccelerationStructureKHR(device, as, 0);
}

void Vulkan::DestroyPool(VkDescriptorPool pool) { vkDestroyDescriptorPool(device, pool, 0); }

int Vulkan::BindlessIndex(GPUImage *image)
{
    BindlessDescriptorPool &descriptorPool = bindlessDescriptorPools[0];
    int index                              = descriptorPool.Allocate();

    Assert(index != -1);

    VkDescriptorImageInfo info = {};
    info.imageView             = image->imageView;
    info.imageLayout           = image->lastLayout;

    VkWriteDescriptorSet writeSet = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeSet.dstSet               = descriptorPool.set;
    writeSet.descriptorCount      = 1;
    writeSet.dstArrayElement      = index;
    writeSet.descriptorType       = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writeSet.pImageInfo           = &info;

    vkUpdateDescriptorSets(device, 1, &writeSet, 0, 0);
    return index;
}

int Vulkan::BindlessIndex(GPUImage *image, VkImageLayout layout, int subresourceIndex)
{
    BindlessDescriptorPool &descriptorPool = bindlessDescriptorPools[0];
    int index                              = descriptorPool.Allocate();

    VkDescriptorImageInfo info = {};
    info.imageView             = subresourceIndex == -1 ? image->imageView
                                                        : image->subresources[subresourceIndex].imageView;
    info.imageLayout           = layout;

    VkWriteDescriptorSet writeSet = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeSet.dstSet               = descriptorPool.set;
    writeSet.descriptorCount      = 1;
    writeSet.dstArrayElement      = index;
    writeSet.descriptorType       = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writeSet.pImageInfo           = &info;

    vkUpdateDescriptorSets(device, 1, &writeSet, 0, 0);
    return index;
}

int Vulkan::BindlessStorageIndex(GPUImage *image, int subresourceIndex)
{
    BindlessDescriptorPool &descriptorPool = bindlessDescriptorPools[2];
    int index                              = descriptorPool.Allocate();

    VkDescriptorImageInfo info = {};
    info.imageView             = subresourceIndex == -1 ? image->imageView
                                                        : image->subresources[subresourceIndex].imageView;
    info.imageLayout           = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writeSet = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeSet.dstSet               = descriptorPool.set;
    writeSet.descriptorCount      = 1;
    writeSet.dstArrayElement      = index;
    writeSet.descriptorType       = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeSet.pImageInfo           = &info;

    vkUpdateDescriptorSets(device, 1, &writeSet, 0, 0);
    return index;
}

int Vulkan::BindlessStorageIndex(GPUBuffer *buffer, size_t offset, size_t range)
{
    BindlessDescriptorPool &descriptorPool = bindlessDescriptorPools[1];
    int index                              = descriptorPool.Allocate();

    VkDescriptorBufferInfo info = {};
    info.buffer                 = buffer->buffer;
    info.offset                 = offset;
    info.range                  = range;

    VkWriteDescriptorSet writeSet = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeSet.dstSet               = descriptorPool.set;
    writeSet.descriptorCount      = 1;
    writeSet.dstArrayElement      = index;
    writeSet.descriptorType       = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeSet.pBufferInfo          = &info;

    vkUpdateDescriptorSets(device, 1, &writeSet, 0, 0);
    return index;
}

void Vulkan::FreeBindlessStorageIndex(int index)
{
    BindlessDescriptorPool &descriptorPool = bindlessDescriptorPools[1];
    descriptorPool.Free(index);
}

u64 Vulkan::GetMinAlignment(VkBufferUsageFlags flags)
{
    u64 minAlignment = 0;
    if ((flags & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0)
    {
        minAlignment = Max(deviceProperties.properties.limits.minStorageBufferOffsetAlignment,
                           minAlignment);
    }
    if ((flags & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0)
    {
        minAlignment = Max(deviceProperties.properties.limits.minUniformBufferOffsetAlignment,
                           minAlignment);
    }
    if ((flags & (VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT ||
                  VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT)) != 0)
    {
        minAlignment = Max(deviceProperties.properties.limits.minTexelBufferOffsetAlignment,
                           minAlignment);
    }
    if ((flags & (VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR ||
                  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR)) != 0)
    {
        // TODO: get from properties?
        minAlignment = Max(minAlignment, 256ull);
    }
    Assert(minAlignment != 0);

    return minAlignment;
}

TransferBuffer Vulkan::GetStagingBuffer(VkBufferUsageFlags flags, size_t totalSize,
                                        int numRanges)
{
    GPUBuffer buffer  = CreateBuffer(flags | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                     totalSize);
    buffer.lastStage  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    buffer.lastAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;

    GPUBuffer stagingBuffer =
        CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, totalSize, MemoryUsage::CPU_TO_GPU);

    void *mappedPtr = stagingBuffer.allocation->GetMappedData();

    TransferBuffer transferBuffer;
    transferBuffer.buffer        = buffer;
    transferBuffer.stagingBuffer = stagingBuffer;
    transferBuffer.mappedPtr     = mappedPtr;
    return transferBuffer;
}

TransferBuffer Vulkan::GetStagingImage(ImageDesc desc)

{
    desc.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    TransferBuffer buffer;
    size_t size  = desc.width * desc.height * GetFormatSize(desc.format);
    buffer.image = CreateImage(desc);

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, buffer.image.image, &req);

    buffer.stagingBuffer =
        CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, req.size, MemoryUsage::CPU_TO_GPU);
    buffer.mappedPtr = buffer.stagingBuffer.allocation->GetMappedData();
    return buffer;
}

TransferBuffer Vulkan::GetReadbackBuffer(VkBufferUsageFlags flags, size_t totalSize)
{
    GPUBuffer buffer = CreateBuffer(flags | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, totalSize);

    GPUBuffer stagingBuffer =
        CreateBuffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, totalSize, MemoryUsage::GPU_TO_CPU);

    void *mappedPtr = stagingBuffer.allocation->GetMappedData();

    TransferBuffer transferBuffer;
    transferBuffer.buffer        = buffer;
    transferBuffer.stagingBuffer = stagingBuffer;
    transferBuffer.mappedPtr     = mappedPtr;
    return transferBuffer;
}

void CommandBuffer::WaitOn(CommandBuffer *other)
{
    Semaphore s   = device->CreateSemaphore();
    s.signalValue = 1;
    other->Signal(s);
    Wait(s);
}

void CommandBuffer::CopyBuffer(GPUBuffer *dst, GPUBuffer *src)
{
    VkBufferCopy bufferCopy = {};
    bufferCopy.srcOffset    = 0;
    bufferCopy.dstOffset    = 0;
    bufferCopy.size         = dst->size;

    vkCmdCopyBuffer(buffer, src->buffer, dst->buffer, 1, &bufferCopy);
}

void CommandBuffer::SubmitTransfer(TransferBuffer *transferBuffer, u32 dstOffset)
{
    VkBufferCopy bufferCopy = {};
    bufferCopy.srcOffset    = 0;
    bufferCopy.dstOffset    = dstOffset;
    bufferCopy.size         = transferBuffer->buffer.size;

    vkCmdCopyBuffer(buffer, transferBuffer->stagingBuffer.buffer,
                    transferBuffer->buffer.buffer, 1, &bufferCopy);

    transferBuffer->buffer.lastStage  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    transferBuffer->buffer.lastAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;
}

TransferBuffer CommandBuffer::SubmitImage(void *ptr, ImageDesc desc)
{
    TransferBuffer transferBuffer = device->GetStagingImage(desc);
    size_t size                   = desc.width * desc.height * GetFormatSize(desc.format);
    Assert(transferBuffer.mappedPtr);
    MemoryCopy(transferBuffer.mappedPtr, ptr, size);

    VkImageMemoryBarrier2 barrier = device->ImageMemoryBarrier(
        transferBuffer.image.image, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_NONE, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);

    VkDependencyInfo info        = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    info.imageMemoryBarrierCount = 1;
    info.pImageMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(buffer, &info);

    VkBufferImageCopy copy           = {};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent                 = {desc.width, desc.height, 1};

    vkCmdCopyBufferToImage(buffer, transferBuffer.stagingBuffer.buffer,
                           transferBuffer.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &copy);
    transferBuffer.image.lastPipeline = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    transferBuffer.image.lastAccess   = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    transferBuffer.image.lastLayout   = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    return transferBuffer;
}

void CommandBuffer::CopyBuffer(GPUBuffer *dst, GPUBuffer *src, BufferToBufferCopy *copies,
                               u32 num)
{
    ScratchArena scratch;
    VkBufferCopy *vkCopies = PushArrayNoZero(scratch.temp.arena, VkBufferCopy, num);
    for (u32 i = 0; i < num; i++)
    {
        VkBufferCopy &copy = vkCopies[i];
        copy.srcOffset     = copies[i].srcOffset;
        copy.dstOffset     = copies[i].dstOffset;
        copy.size          = copies[i].size;
    }

    vkCmdCopyBuffer(buffer, src->buffer, dst->buffer, num, vkCopies);
}

void CommandBuffer::CopyImage(GPUBuffer *transfer, GPUImage *image, BufferImageCopy *copies,
                              u32 num)
{
    ScratchArena scratch;
    VkBufferImageCopy *vkCopies = PushArrayNoZero(scratch.temp.arena, VkBufferImageCopy, num);
    for (int i = 0; i < num; i++)
    {
        VkBufferImageCopy &copy              = vkCopies[i];
        copy.bufferOffset                    = copies[i].bufferOffset;
        copy.bufferRowLength                 = copies[i].rowLength;
        copy.bufferImageHeight               = copies[i].imageHeight;
        copy.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.mipLevel       = copies[i].mipLevel;
        copy.imageSubresource.baseArrayLayer = copies[i].baseLayer;
        copy.imageSubresource.layerCount     = copies[i].layerCount;
        copy.imageOffset.x                   = copies[i].offset.x;
        copy.imageOffset.y                   = copies[i].offset.y;
        copy.imageOffset.z                   = copies[i].offset.z;
        copy.imageExtent.width               = copies[i].extent.x;
        copy.imageExtent.height              = copies[i].extent.y;
        copy.imageExtent.depth               = copies[i].extent.z;
    }
    vkCmdCopyBufferToImage(buffer, transfer->buffer, image->image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, num, vkCopies);
}

void CommandBuffer::CopyImage(GPUImage *dst, GPUImage *src, const ImageToImageCopy &copy)
{
    VkImageCopy imageCopy;
    imageCopy.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    imageCopy.srcSubresource.mipLevel       = copy.srcMipLevel;
    imageCopy.srcSubresource.baseArrayLayer = copy.srcBaseLayer;
    imageCopy.srcSubresource.layerCount     = copy.srcLayerCount;
    imageCopy.srcOffset.x                   = copy.srcOffset.x;
    imageCopy.srcOffset.y                   = copy.srcOffset.y;
    imageCopy.srcOffset.z                   = copy.srcOffset.z;

    imageCopy.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    imageCopy.dstSubresource.mipLevel       = copy.dstMipLevel;
    imageCopy.dstSubresource.baseArrayLayer = copy.dstBaseLayer;
    imageCopy.dstSubresource.layerCount     = copy.dstLayerCount;
    imageCopy.dstOffset.x                   = copy.dstOffset.x;
    imageCopy.dstOffset.y                   = copy.dstOffset.y;
    imageCopy.dstOffset.z                   = copy.dstOffset.z;

    imageCopy.extent.width  = copy.extent.x;
    imageCopy.extent.height = copy.extent.y;
    imageCopy.extent.depth  = copy.extent.z;

    vkCmdCopyImage(buffer, src->image, src->lastLayout, dst->image, dst->lastLayout, 1,
                   &imageCopy);
}

void CommandBuffer::CopyImageToBuffer(GPUBuffer *dst, GPUImage *src,
                                      const BufferImageCopy *copies, u32 num)
{
    ScratchArena scratch;
    VkBufferImageCopy *vkCopies = PushArrayNoZero(scratch.temp.arena, VkBufferImageCopy, num);
    for (int i = 0; i < num; i++)
    {
        VkBufferImageCopy &copy              = vkCopies[i];
        copy.bufferOffset                    = copies[i].bufferOffset;
        copy.bufferRowLength                 = copies[i].rowLength;
        copy.bufferImageHeight               = copies[i].imageHeight;
        copy.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.mipLevel       = copies[i].mipLevel;
        copy.imageSubresource.baseArrayLayer = copies[i].baseLayer;
        copy.imageSubresource.layerCount     = copies[i].layerCount;
        copy.imageOffset.x                   = copies[i].offset.x;
        copy.imageOffset.y                   = copies[i].offset.y;
        copy.imageOffset.z                   = copies[i].offset.z;
        copy.imageExtent.width               = copies[i].extent.x;
        copy.imageExtent.height              = copies[i].extent.y;
        copy.imageExtent.depth               = copies[i].extent.z;
    }
    Assert(src->lastLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdCopyImageToBuffer(buffer, src->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           dst->buffer, num, vkCopies);
}

TransferBuffer CommandBuffer::SubmitBuffer(void *ptr, VkBufferUsageFlags2 flags,
                                           size_t totalSize, u32 dstOffset)
{
    TransferBuffer transferBuffer = device->GetStagingBuffer(flags, totalSize);
    Assert(transferBuffer.mappedPtr);
    MemoryCopy(transferBuffer.mappedPtr, ptr, totalSize);
    SubmitTransfer(&transferBuffer, dstOffset);
    return transferBuffer;
}

void CommandBuffer::SubmitBuffer(GPUBuffer *dst, void *ptr, size_t totalSize, u32 dstOffset)
{
    GPUBuffer stagingBuffer = device->CreateBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, totalSize,
                                                   MemoryUsage::CPU_TO_GPU);
    void *mappedPtr         = stagingBuffer.allocation->GetMappedData();

    Assert(mappedPtr);
    MemoryCopy(mappedPtr, ptr, totalSize);

    VkBufferCopy bufferCopy = {};
    bufferCopy.srcOffset    = 0;
    bufferCopy.dstOffset    = dstOffset;
    bufferCopy.size         = totalSize;

    vkCmdCopyBuffer(buffer, stagingBuffer.buffer, dst->buffer, 1, &bufferCopy);

    dst->lastStage  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    dst->lastAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;
}

void CommandBuffer::BindPipeline(VkPipelineBindPoint bindPoint, VkPipeline pipeline)
{
    vkCmdBindPipeline(buffer, bindPoint, pipeline);
}

int DescriptorSetLayout::AddBinding(u32 b, DescriptorType type, VkShaderStageFlags stage)
{
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding                      = b;
    binding.descriptorType               = ConvertDescriptorType(type);
    binding.descriptorCount              = 1;
    binding.stageFlags                   = stage;

    int index = static_cast<int>(bindings.size());
    bindings.push_back(binding);
    return index;
}

void DescriptorSetLayout::AddImmutableSamplers()
{
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding                      = 50;
    binding.descriptorType               = VK_DESCRIPTOR_TYPE_SAMPLER;
    binding.descriptorCount              = 1;
    binding.stageFlags                   = VK_SHADER_STAGE_ALL;
    binding.pImmutableSamplers           = &device->immutableSamplers[0];

    VkDescriptorSetLayoutBinding binding2 = {};
    binding2.binding                      = 51;
    binding2.descriptorType               = VK_DESCRIPTOR_TYPE_SAMPLER;
    binding2.descriptorCount              = 1;
    binding2.stageFlags                   = VK_SHADER_STAGE_ALL;
    binding2.pImmutableSamplers           = &device->immutableSamplers[1];

    bindings.push_back(binding);
    bindings.push_back(binding2);
}

VkDescriptorSetLayout *DescriptorSetLayout::GetVulkanLayout()
{
    if (layout == VK_NULL_HANDLE)
    {
        ScratchArena scratch;

        Assert(bindings.size());
        VkDescriptorSetLayoutCreateInfo createInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        createInfo.bindingCount = bindings.size();
        createInfo.pBindings    = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device->device, &createInfo, 0, &layout));
    }
    return &layout;
}

DescriptorSet &DescriptorSet::Bind(int index, GPUImage *image, int subresource)
{
    Assert(index < layout->bindings.size() && index < descriptorInfo.size());
    VkDescriptorImageInfo info = {};
    info.imageView =
        subresource == -1 ? image->imageView : image->subresources[subresource].imageView;
    info.imageLayout = image->lastLayout;

    descriptorInfo[index].image = info;

    VkWriteDescriptorSet writeSet = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeSet.descriptorType       = layout->bindings[index].descriptorType;
    writeSet.descriptorCount      = 1;
    writeSet.dstSet               = VK_NULL_HANDLE;
    writeSet.dstBinding           = layout->bindings[index].binding;
    writeSet.pImageInfo           = &descriptorInfo[index].image;

    writeDescriptorSets.push_back(writeSet);
    return *this;
}

DescriptorSet &DescriptorSet::Bind(int index, GPUBuffer *buffer, u64 offset, u64 size)
{
    Assert(index < layout->bindings.size() && index < descriptorInfo.size());
    VkDescriptorBufferInfo info = {};
    info.buffer                 = buffer->buffer;
    info.offset                 = offset;
    info.range                  = size;

    descriptorInfo[index].buffer = info;

    VkWriteDescriptorSet writeSet = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeSet.descriptorType       = layout->bindings[index].descriptorType;
    writeSet.descriptorCount      = 1;
    writeSet.dstSet               = VK_NULL_HANDLE;
    writeSet.dstBinding           = layout->bindings[index].binding;
    writeSet.pBufferInfo          = &descriptorInfo[index].buffer;

    writeDescriptorSets.push_back(writeSet);
    return *this;
}

DescriptorSet &DescriptorSet::Bind(int index, VkAccelerationStructureKHR *accel)
{
    Assert(index < layout->bindings.size() && index < descriptorInfo.size());
    VkWriteDescriptorSetAccelerationStructureKHR info = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    info.pAccelerationStructures    = accel;
    info.accelerationStructureCount = 1;

    descriptorInfo[index].accel = info;

    VkWriteDescriptorSet writeSet = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeSet.pNext                = &descriptorInfo[index].accel;
    writeSet.descriptorType       = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writeSet.descriptorCount      = 1;
    writeSet.dstSet               = VK_NULL_HANDLE;
    writeSet.dstBinding           = layout->bindings[index].binding;

    writeDescriptorSets.push_back(writeSet);
    return *this;
}

DescriptorSet &DescriptorSet::Bind(int index, u64 *ptlasAddress)
{
    VkWriteDescriptorSetPartitionedAccelerationStructureNV info = {
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_PARTITIONED_ACCELERATION_STRUCTURE_NV};
    info.pAccelerationStructures    = ptlasAddress;
    info.accelerationStructureCount = 1;

    descriptorInfo[index].ptlas = info;

    VkWriteDescriptorSet writeSet = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writeSet.pNext                = &descriptorInfo[index].ptlas;
    writeSet.descriptorType       = VK_DESCRIPTOR_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_NV;
    writeSet.descriptorCount      = 1;
    writeSet.dstSet               = VK_NULL_HANDLE;
    writeSet.dstBinding           = layout->bindings[index].binding;

    writeDescriptorSets.push_back(writeSet);
    return *this;
}

DescriptorSet &DescriptorSet::Bind(GPUBuffer *buffer, u64 offset, u64 size)
{
    u32 index = numBinds++;
    return Bind(index, buffer, offset, size);
}

DescriptorSet &DescriptorSet::Bind(GPUImage *image, int subresource)
{
    u32 index = numBinds++;
    return Bind(index, image, subresource);
}

DescriptorSet &DescriptorSet::Bind(VkAccelerationStructureKHR *accel)
{
    u32 index = numBinds++;
    return Bind(index, accel);
}

DescriptorSet &DescriptorSet::Bind(u64 *ptlasAddress)
{
    u32 index = numBinds++;
    return Bind(index, ptlasAddress);
}

void CommandBuffer::BindDescriptorSets(VkPipelineBindPoint bindPoint, DescriptorSet *set,
                                       VkPipelineLayout pipeLayout)
{
    for (auto &write : set->writeDescriptorSets)
    {
        write.dstSet = set->set;
    }
    vkUpdateDescriptorSets(device->device, set->writeDescriptorSets.size(),
                           set->writeDescriptorSets.data(), 0, 0);
    vkCmdBindDescriptorSets(buffer, bindPoint, pipeLayout, 0, 1, &set->set, 0, 0);
    vkCmdBindDescriptorSets(buffer, bindPoint, pipeLayout, 1,
                            device->bindlessDescriptorSets.size(),
                            device->bindlessDescriptorSets.data(), 0, 0);
    set->writeDescriptorSets.clear();
}

ResourceBinding CommandBuffer::StartBinding(VkPipelineBindPoint bindPoint, VkPipeline pipeline,
                                            DescriptorSetLayout *layout)
{
    BindPipeline(bindPoint, pipeline);

    ResourceBinding binding;
    binding.bindPoint = bindPoint;
    binding.ds        = layout->CreateDescriptorSet();
    binding.cmd       = this;
    return binding;
}

ResourceBinding CommandBuffer::StartBindingCompute(VkPipeline pipeline,
                                                   DescriptorSetLayout *layout)
{
    return StartBinding(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline, layout);
}

ResourceBinding &ResourceBinding::PushConstants(PushConstant *push, void *ptr)
{
    cmd->PushConstants(push, ptr, ds.layout->pipelineLayout);
    return *this;
}

void ResourceBinding::End()
{
    cmd->BindDescriptorSets(bindPoint, &ds, ds.layout->pipelineLayout);
}

void CommandBuffer::TraceRays(RayTracingState *state, u32 width, u32 height, u32 depth)
{
    vkCmdTraceRaysKHR(buffer, &state->raygen, &state->miss, &state->hit, &state->call, width,
                      height, depth);
}

void CommandBuffer::TraceRaysIndirect(RayTracingState *state, GPUBuffer *indirect, u32 offset)
{
    u64 indirectAddress = device->GetDeviceAddress(indirect) + offset;
    vkCmdTraceRaysIndirectKHR(buffer, &state->raygen, &state->miss, &state->hit, &state->call,
                              indirectAddress);
}

void CommandBuffer::Dispatch(u32 groupCountX, u32 groupCountY, u32 groupCountZ)
{
    vkCmdDispatch(buffer, groupCountX, groupCountY, groupCountZ);
}

void CommandBuffer::DispatchIndirect(GPUBuffer *gpuBuffer, u32 offset)
{
    vkCmdDispatchIndirect(buffer, gpuBuffer->buffer, offset);
}

u64 Vulkan::GetDeviceAddress(VkBuffer buffer)
{
    VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer                    = buffer;

    return vkGetBufferDeviceAddress(device, &info);
}

u64 Vulkan::GetDeviceAddress(GPUBuffer *buffer)
{
    VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer                    = buffer->buffer;

    return vkGetBufferDeviceAddress(device, &info);
}

void Vulkan::SetName(u64 handle, VkObjectType type, const char *name)
{
    if (!debugUtils || handle == 0)
    {
        return;
    }
    VkDebugUtilsObjectNameInfoEXT info = {};
    info.sType                         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.pObjectName                   = name;
    info.objectType                    = type;
    info.objectHandle                  = handle;
    VK_CHECK(vkSetDebugUtilsObjectNameEXT(device, &info));
}

void Vulkan::SetName(VkDescriptorSetLayout handle, const char *name)
{
    SetName((u64)handle, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, name);
}

void Vulkan::SetName(VkDescriptorSet handle, const char *name)
{
    SetName((u64)handle, VK_OBJECT_TYPE_DESCRIPTOR_SET, name);
}

void Vulkan::SetName(VkShaderModule handle, const char *name)
{
    SetName((u64)handle, VK_OBJECT_TYPE_SHADER_MODULE, name);
}

void Vulkan::SetName(VkPipeline handle, const char *name)
{
    SetName((u64)handle, VK_OBJECT_TYPE_PIPELINE, name);
}

void Vulkan::SetName(VkPipelineLayout handle, const char *name)
{
    SetName((u64)handle, VK_OBJECT_TYPE_PIPELINE_LAYOUT, name);
}

void Vulkan::SetName(VkQueue handle, const char *name)
{
    SetName((u64)handle, VK_OBJECT_TYPE_QUEUE, name);
}

void Vulkan::SetName(GPUBuffer *buffer, const char *name)
{
    SetName((u64)buffer->buffer, VK_OBJECT_TYPE_BUFFER, name);
}

void Vulkan::SetName(VkCommandBuffer handle, const char *name)
{
    SetName((u64)handle, VK_OBJECT_TYPE_COMMAND_BUFFER, name);
}

void Vulkan::BeginEvent(CommandBuffer *cmd, string name)
{
    if (!debugUtils) return;

    VkDebugUtilsLabelEXT label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName           = (const char *)name.str;
    label.color[0]             = 0.f;
    label.color[1]             = 0.f;
    label.color[2]             = 0.f;
    label.color[3]             = 0.f;

    vkCmdBeginDebugUtilsLabelEXT(cmd->buffer, &label);
}

void Vulkan::EndEvent(CommandBuffer *cmd) { vkCmdEndDebugUtilsLabelEXT(cmd->buffer); }

Shader Vulkan::CreateShader(ShaderStage stage, string name, string shaderData)
{
    Shader shader;
    shader.stage                        = stage;
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType                    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode                    = (u32 *)shaderData.str;
    createInfo.codeSize                 = shaderData.size;
    VK_CHECK(vkCreateShaderModule(device, &createInfo, 0, &shader.module));

    if (name.size) SetName(shader.module, (const char *)name.str);
    return shader;
}

void DescriptorSetLayout::CreatePipelineLayout(PushConstant *pc)
{
    std::vector<VkDescriptorSetLayout> layouts;
    layouts.reserve(1 + device->bindlessDescriptorSetLayouts.size());
    layouts.push_back(*GetVulkanLayout());
    for (auto &bindlessLayout : device->bindlessDescriptorSetLayouts)
    {
        layouts.push_back(bindlessLayout);
    }

    VkPipelineLayoutCreateInfo createInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    createInfo.setLayoutCount             = layouts.size();
    createInfo.pSetLayouts                = layouts.data();

    VkPushConstantRange vkpc;
    if (pc)
    {
        vkpc.offset                       = pc->offset;
        vkpc.size                         = pc->size;
        vkpc.stageFlags                   = GetVulkanShaderStage(pc->stage);
        createInfo.pushConstantRangeCount = 1;
        createInfo.pPushConstantRanges    = &vkpc;
    }
    VK_CHECK(vkCreatePipelineLayout(device->device, &createInfo, 0, &pipelineLayout));
}

VkPipeline Vulkan::CreateComputePipeline(Shader *shader, DescriptorSetLayout *layout,
                                         PushConstant *pc, string name)
{
    VkPipelineShaderStageCreateInfo pipelineInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.pName  = "main";
    pipelineInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.module = shader->module;

    if (layout->pipelineLayout == VK_NULL_HANDLE) layout->CreatePipelineLayout(pc);

    VkComputePipelineCreateInfo createInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    createInfo.stage                       = pipelineInfo;
    createInfo.layout                      = layout->pipelineLayout;

    VkPipeline pipeline;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &createInfo, 0, &pipeline);
    if (name.size) SetName(pipeline, (const char *)name.str);
    return pipeline;
}

RayTracingState Vulkan::CreateRayTracingPipeline(RayTracingShaderGroup *shaderGroups,
                                                 int numShaderGroups, PushConstant *pc,
                                                 DescriptorSetLayout *layout, u32 maxDepth,
                                                 bool clusters)
{
    int total = 0;
    for (int i = 0; i < numShaderGroups; i++)
    {
        total += shaderGroups[i].numShaders;
    }

    ScratchArena scratch;
    VkPipelineShaderStageCreateInfo *pipelineInfos =
        PushArrayNoZero(scratch.temp.arena, VkPipelineShaderStageCreateInfo, total);
    VkRayTracingShaderGroupCreateInfoKHR *vkShaderGroups = PushArrayNoZero(
        scratch.temp.arena, VkRayTracingShaderGroupCreateInfoKHR, numShaderGroups);

    int count = 0;
    // Create pipeline infos
    {
        VkPipelineShaderStageCreateInfo info = {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.pName = "main";

        for (int i = 0; i < numShaderGroups; i++)
        {
            RayTracingShaderGroup &shaderGroup = shaderGroups[i];
            for (int j = 0; j < shaderGroup.numShaders; j++)
            {
                switch (shaderGroup.stage[j])
                {
                    case ShaderStage::Raygen:
                        info.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
                        break;
                    case ShaderStage::Miss: info.stage = VK_SHADER_STAGE_MISS_BIT_KHR; break;
                    case ShaderStage::Hit:
                        info.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
                        break;
                    case ShaderStage::Intersect:
                        info.stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
                        break;
                    default: Assert(0);
                }
                info.module            = shaderGroup.shaders[j].module;
                pipelineInfos[count++] = info;
            }
        }
    }
    count           = 0;
    int shaderCount = 0;

    int counts[RST_Max] = {};
    // Create shader groups
    {
        for (int i = 0; i < numShaderGroups; i++)
        {
            VkRayTracingShaderGroupCreateInfoKHR group = {
                VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
            group.type                      = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            group.generalShader             = VK_SHADER_UNUSED_KHR;
            group.closestHitShader          = VK_SHADER_UNUSED_KHR;
            group.anyHitShader              = VK_SHADER_UNUSED_KHR;
            group.intersectionShader        = VK_SHADER_UNUSED_KHR;
            RayTracingShaderGroup &rtsGroup = shaderGroups[i];

            VkRayTracingShaderGroupTypeKHR vkShaderGroupType =
                rtsGroup.type == RayTracingShaderGroupType::Triangle
                    ? VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR
                    : VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;

            for (int j = 0; j < rtsGroup.numShaders; j++)
            {
                switch (rtsGroup.stage[j])
                {
                    case ShaderStage::Raygen:
                    {
                        Assert(rtsGroup.numShaders == 1);
                        group.generalShader = count;
                        shaderCount++;
                        counts[RST_Raygen]++;
                    }
                    break;
                    case ShaderStage::Miss:
                    {
                        Assert(rtsGroup.numShaders == 1);
                        group.generalShader = count;
                        shaderCount++;
                        counts[RST_Miss]++;
                    }
                    break;
                    case ShaderStage::Hit:
                    {
                        group.type             = vkShaderGroupType;
                        group.closestHitShader = shaderCount++;
                        counts[RST_Hit]++;
                    }
                    break;
                    case ShaderStage::Intersect:
                    {
                        group.type               = vkShaderGroupType;
                        group.intersectionShader = shaderCount++;
                        counts[RST_Intersect]++;
                    }
                    break;
                    default: Assert(0);
                }
            }
            vkShaderGroups[count++] = group;
        }
    }

    // Create pipeline layout
    if (layout->pipelineLayout == VK_NULL_HANDLE) layout->CreatePipelineLayout(pc);
    VkPipelineLayout pipelineLayout = layout->pipelineLayout;

    // Create pipeline
    VkPipeline pipeline;
    {
        VkRayTracingPipelineCreateInfoKHR pipelineInfo = {
            VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};

        pipelineInfo.pStages                      = pipelineInfos;
        pipelineInfo.stageCount                   = total;
        pipelineInfo.pGroups                      = vkShaderGroups;
        pipelineInfo.groupCount                   = numShaderGroups;
        pipelineInfo.maxPipelineRayRecursionDepth = maxDepth;
        pipelineInfo.layout                       = pipelineLayout;

        VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV pipeClusters = {
            VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV};

        if (clusters)
        {
            pipelineInfo.pNext                             = &pipeClusters;
            pipeClusters.allowClusterAccelerationStructure = true;
        }

        VK_CHECK(
            vkCreateRayTracingPipelinesKHR(device, {}, {}, 1, &pipelineInfo, 0, &pipeline));
    }

    RayTracingState state = {};
    state.pipeline        = pipeline;
    state.layout          = pipelineLayout;

    // Create shader binding table
    {
        Assert(IsPow2(rtPipeProperties.shaderGroupBaseAlignment));
        Assert(IsPow2(rtPipeProperties.shaderGroupHandleAlignment));
        u32 handleSize = rtPipeProperties.shaderGroupHandleSize;
        u32 alignedHandleSize =
            AlignPow2(handleSize, rtPipeProperties.shaderGroupHandleAlignment);

        FixedArray<int, RST_Max> offsets = {0, 0, 0, 0};
        int offset                       = 0;
        for (int i = 0; i < RST_Max; i++)
        {
            state.addresses[i].size   = AlignPow2(counts[i] * alignedHandleSize,
                                                  rtPipeProperties.shaderGroupBaseAlignment);
            state.addresses[i].stride = state.addresses[i].size;
            offsets[i]                = offset;
            offset += state.addresses[i].size;
        }

        u32 dataSize = handleSize * total;

        u8 *data = PushArrayNoZero(arena, u8, dataSize);
        VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, 0, numShaderGroups,
                                                      dataSize, data));

        GPUBuffer buffer = CreateBuffer(VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                        offset, MemoryUsage::GPU_TO_CPU);

        VkBufferDeviceAddressInfo deviceAddressInfo = {
            VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        deviceAddressInfo.buffer   = buffer.buffer;
        VkDeviceAddress sbtAddress = vkGetBufferDeviceAddress(device, &deviceAddressInfo);

        u8 *base = (u8 *)buffer.allocation->GetMappedData();
        for (int type = 0; type < RST_Max; type++)
        {
            if (state.addresses[type].size)
            {
                state.addresses[type].deviceAddress = sbtAddress;
                sbtAddress += state.addresses[type].size;
                u8 *ptr = base + offsets[type];
                for (int i = 0; i < counts[type]; i++)
                {
                    MemoryCopy(ptr, data, handleSize);
                    data += handleSize;
                    ptr += alignedHandleSize;
                }
            }
        }
    }
    return state;
}

DescriptorSet DescriptorSetLayout::CreateDescriptorSet()
{
    ThreadPool &pool                         = device->GetThreadPool(GetThreadIndex());
    VkDescriptorPool descriptorPool          = pool.descriptorPool[device->GetCurrentBuffer()];
    VkDescriptorSetAllocateInfo allocateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocateInfo.descriptorPool     = descriptorPool;
    allocateInfo.pSetLayouts        = GetVulkanLayout();
    allocateInfo.descriptorSetCount = 1;

    DescriptorSet set;
    VkResult result = vkAllocateDescriptorSets(device->device, &allocateInfo, &set.set);
    ErrorExit(result == VK_SUCCESS, "Error while allocating descriptor sets: %u\n", result);
    set.layout = this;
    set.descriptorInfo.resize(bindings.size());
    return set;
}

DescriptorSet DescriptorSetLayout::CreateNewDescriptorSet()
{
    ScratchArena scratch;

    VkDescriptorPoolCreateInfo poolCreateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};

    StaticArray<VkDescriptorPoolSize> poolSizes(scratch.temp.arena, bindings.size());

    for (auto &binding : bindings)
    {
        bool unique = true;
        for (auto &poolSize : poolSizes)
        {
            if (poolSize.type == binding.descriptorType)
            {
                poolSize.descriptorCount++;
                unique = false;
                break;
            }
        }
        if (unique)
        {
            VkDescriptorPoolSize poolSize;
            poolSize.type            = binding.descriptorType;
            poolSize.descriptorCount = 1;
            poolSizes.Push(poolSize);
        }
    }

    poolCreateInfo.pPoolSizes    = poolSizes.data;
    poolCreateInfo.poolSizeCount = poolSizes.Length();
    poolCreateInfo.maxSets       = 1;

    VkDescriptorPool pool;
    vkCreateDescriptorPool(device->device, &poolCreateInfo, 0, &pool);

    VkDescriptorSetAllocateInfo allocateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocateInfo.descriptorPool     = pool;
    allocateInfo.pSetLayouts        = GetVulkanLayout();
    allocateInfo.descriptorSetCount = 1;

    DescriptorSet set;
    VkResult result = vkAllocateDescriptorSets(device->device, &allocateInfo, &set.set);
    ErrorExit(result == VK_SUCCESS, "Error while allocating descriptor sets: %u\n", result);
    set.pool   = pool;
    set.layout = this;
    set.descriptorInfo.resize(bindings.size());
    return set;
}

void DescriptorSet::Reset()
{
    VK_CHECK(vkResetDescriptorPool(device->device, pool, 0));
    VkDescriptorSetAllocateInfo allocateInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocateInfo.descriptorPool     = pool;
    allocateInfo.pSetLayouts        = layout->GetVulkanLayout();
    allocateInfo.descriptorSetCount = 1;

    VkResult result = vkAllocateDescriptorSets(device->device, &allocateInfo, &set);
    ErrorExit(result == VK_SUCCESS, "Error while allocating descriptor sets: %u\n", result);
}

GPUAccelerationStructurePayload CommandBuffer::BuildAS(
    VkAccelerationStructureTypeKHR accelType, VkAccelerationStructureGeometryKHR *geometries,
    int count, VkAccelerationStructureBuildRangeInfoKHR *buildRanges, u32 *maxPrimitiveCounts)
{
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type  = accelType;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    buildInfo.pGeometries   = geometries;
    buildInfo.geometryCount = count;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};

    vkGetAccelerationStructureBuildSizesKHR(device->device,
                                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &buildInfo, maxPrimitiveCounts, &sizeInfo);

    GPUBuffer scratch = device->CreateBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                             sizeInfo.buildScratchSize);

    VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer                    = scratch.buffer;
    VkDeviceAddress scratchAddress = vkGetBufferDeviceAddress(device->device, &info);

    GPUAccelerationStructurePayload payload;
    payload.as.type = accelType;
    payload.scratch = scratch;

    {
        payload.as.buffer =
            device->CreateBuffer(VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                 sizeInfo.accelerationStructureSize);
        VkAccelerationStructureCreateInfoKHR accelCreateInfo = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};

        accelCreateInfo.buffer = payload.as.buffer.buffer;
        accelCreateInfo.size   = sizeInfo.accelerationStructureSize;
        accelCreateInfo.type   = accelType;

        vkCreateAccelerationStructureKHR(device->device, &accelCreateInfo, 0, &payload.as.as);
    }

    buildInfo.scratchData.deviceAddress = scratchAddress;
    buildInfo.dstAccelerationStructure  = payload.as.as;

    VkAccelerationStructureDeviceAddressInfoKHR accelDeviceAddressInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    accelDeviceAddressInfo.accelerationStructure = payload.as.as;
    payload.as.address =
        vkGetAccelerationStructureDeviceAddressKHR(device->device, &accelDeviceAddressInfo);

    vkCmdBuildAccelerationStructuresKHR(buffer, 1, &buildInfo, &buildRanges);
    return payload;
}

VkAccelerationStructureKHR CommandBuffer::BuildAS(
    GPUBuffer *accel, GPUBuffer *scratch, VkAccelerationStructureTypeKHR accelType,
    VkAccelerationStructureGeometryKHR *geometries, int count,
    VkAccelerationStructureBuildRangeInfoKHR *buildRanges, u32 *maxPrimitiveCounts)
{
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type  = accelType;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    buildInfo.pGeometries   = geometries;
    buildInfo.geometryCount = count;

    VkDeviceAddress scratchAddress = device->GetDeviceAddress(scratch->buffer);

    VkAccelerationStructureKHR as;

    {
        VkAccelerationStructureCreateInfoKHR accelCreateInfo = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};

        accelCreateInfo.buffer = accel->buffer;
        accelCreateInfo.size   = accel->size;
        accelCreateInfo.type   = accelType;

        vkCreateAccelerationStructureKHR(device->device, &accelCreateInfo, 0, &as);
    }

    buildInfo.scratchData.deviceAddress = scratchAddress;
    buildInfo.dstAccelerationStructure  = as;

    vkCmdBuildAccelerationStructuresKHR(buffer, 1, &buildInfo, &buildRanges);
    return as;
}

GPUAccelerationStructurePayload CommandBuffer::BuildCustomBLAS(GPUBuffer *aabbsBuffer,
                                                               u32 numAabbs)
{
    VkAccelerationStructureGeometryKHR geometry = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    auto &aabbs           = geometry.geometry.aabbs;
    aabbs.sType           = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    aabbs.data.deviceAddress = device->GetDeviceAddress(aabbsBuffer->buffer);
    aabbs.stride             = sizeof(VkAabbPositionsKHR);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
    rangeInfo.primitiveCount                           = numAabbs;

    return BuildAS(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, &geometry, 1, &rangeInfo,
                   &numAabbs);
}

void CommandBuffer::BuildCustomBLAS(StaticArray<AccelBuildInfo> &blasBuildInfos)
{
    ScratchArena scratch;

    StaticArray<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(
        scratch.temp.arena, blasBuildInfos.Length());
    VkAccelerationStructureBuildRangeInfoKHR **buildRangeInfos =
        PushArrayNoZero(scratch.temp.arena, VkAccelerationStructureBuildRangeInfoKHR *,
                        blasBuildInfos.Length());
    Array<VkAccelerationStructureGeometryKHR> geometryDatas(scratch.temp.arena,
                                                            blasBuildInfos.Length());

    for (int blasBuildInfoIndex = 0; blasBuildInfoIndex < blasBuildInfos.Length();
         blasBuildInfoIndex++)
    {
        AccelBuildInfo &blasBuildInfo = blasBuildInfos[blasBuildInfoIndex];
        buildRangeInfos[blasBuildInfoIndex] =
            PushStruct(scratch.temp.arena, VkAccelerationStructureBuildRangeInfoKHR);

        buildRangeInfos[blasBuildInfoIndex]->primitiveCount  = blasBuildInfo.primitiveCount;
        buildRangeInfos[blasBuildInfoIndex]->primitiveOffset = blasBuildInfo.primitiveOffset;

        VkAccelerationStructureGeometryKHR geometry = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;

        auto &aabbs = geometry.geometry.aabbs;
        aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
        aabbs.data.deviceAddress = blasBuildInfo.dataDeviceAddress;
        aabbs.stride             = sizeof(VkAabbPositionsKHR);

        geometryDatas.Push(geometry);

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        buildInfo.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                          VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
        buildInfo.pGeometries               = &geometryDatas[geometryDatas.Length() - 1];
        buildInfo.geometryCount             = 1;
        buildInfo.scratchData.deviceAddress = blasBuildInfo.scratchDataDeviceAddress;
        buildInfo.dstAccelerationStructure  = blasBuildInfo.as;

        buildInfos.Push(buildInfo);
    }

    vkCmdBuildAccelerationStructuresKHR(buffer, blasBuildInfos.Length(), buildInfos.data,
                                        buildRangeInfos);
}

void Vulkan::ConvertCLASIndirectInfo(CLASOpInput opInput, CLASOpType opType,
                                     VkClusterAccelerationStructureInputInfoNV &inputInfo,
                                     VkOpInput &vkOpInput, VkDeviceSize &srcInfosArrayStride)
{
    VkClusterAccelerationStructureOpTypeNV vkOpType;

    if (EnumHasAnyFlags(opType, CLASOpType::Move))
    {
        bool inputType = EnumHasAnyFlags(opType, CLASOpType::CLAS);
        vkOpType       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV;
        vkOpInput.moveObjects.sType =
            VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV;
        vkOpInput.moveObjects.type =
            inputType ? VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_NV
                      : VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_CLUSTERS_BOTTOM_LEVEL_NV;
        vkOpInput.moveObjects.noMoveOverlap = opInput.moveObjects.noMoveOverlap;
        vkOpInput.moveObjects.maxMovedBytes = opInput.moveObjects.maxMovedBytes;

        inputInfo.opInput.pMoveObjects = &vkOpInput.moveObjects;
        srcInfosArrayStride = sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV);
    }
    else if (EnumHasAnyFlags(opType, CLASOpType::CLAS))
    {
        if (EnumHasAnyFlags(opType, CLASOpType::BuildTemplate))
        {
            vkOpType =
                VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
            srcInfosArrayStride =
                sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV);
        }
        else if (EnumHasAnyFlags(opType, CLASOpType::InstantiateTemplate))
        {
            vkOpType =
                VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
            srcInfosArrayStride =
                sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV);
        }
        else
        {
            vkOpType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
            srcInfosArrayStride =
                sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV);
        }
        vkOpInput.triangleClusters.sType =
            VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV;
        vkOpInput.triangleClusters.vertexFormat                  = VK_FORMAT_R32G32B32_SFLOAT;
        vkOpInput.triangleClusters.maxGeometryIndexValue         = 1;
        vkOpInput.triangleClusters.maxClusterUniqueGeometryCount = 1;
        vkOpInput.triangleClusters.maxClusterTriangleCount =
            opInput.triangleClusters.maxClusterTriangles;
        vkOpInput.triangleClusters.maxClusterVertexCount =
            opInput.triangleClusters.maxClusterVertices;
        vkOpInput.triangleClusters.maxTotalTriangleCount =
            opInput.triangleClusters.maxNumTriangles;
        vkOpInput.triangleClusters.maxTotalVertexCount =
            opInput.triangleClusters.maxNumVertices;
        vkOpInput.triangleClusters.minPositionTruncateBitCount = 0;

        inputInfo.opInput.pTriangleClusters = &vkOpInput.triangleClusters;
    }
    else if (EnumHasAnyFlags(opType, CLASOpType::BLAS))
    {
        vkOpType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
        vkOpInput.clustersBottomLevel.sType =
            VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV;
        vkOpInput.clustersBottomLevel.maxTotalClusterCount =
            opInput.clusterBottomLevel.maxTotalClusterCount;
        vkOpInput.clustersBottomLevel.maxClusterCountPerAccelerationStructure =
            opInput.clusterBottomLevel.maxClusterCountPerAccelerationStructure;

        inputInfo.opInput.pClustersBottomLevel = &vkOpInput.clustersBottomLevel;
        srcInfosArrayStride =
            sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV);
    }
    inputInfo.opType                        = vkOpType;
    inputInfo.maxAccelerationStructureCount = opInput.maxAccelerationStructureCount;
}

void CommandBuffer::CLASIndirect(CLASOpInput opInput, CLASOpMode opMode, CLASOpType opType,
                                 GPUBuffer *dstImplicitData, GPUBuffer *scratchBuffer,
                                 GPUBuffer *dstAddresses, GPUBuffer *dstSizes,
                                 GPUBuffer *srcInfosArray, GPUBuffer *srcInfosCount,
                                 u32 srcInfosOffset, u32 dstClasOffset)
{

    VkDeviceSize srcInfosArrayStride;
    VkOpInput vkOpInput = {};
    VkClusterAccelerationStructureOpModeNV vkOpMode;

    switch (opMode)
    {
        case CLASOpMode::ImplicitDestinations:
            vkOpMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
            break;
        case CLASOpMode::ExplicitDestinations:
            vkOpMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
            break;
        case CLASOpMode::ComputeSizes:
            vkOpMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
            break;
    }

    VkClusterAccelerationStructureInputInfoNV inputInfo = {
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputInfo.opMode = vkOpMode;
    inputInfo.flags  = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

    device->ConvertCLASIndirectInfo(opInput, opType, inputInfo, vkOpInput,
                                    srcInfosArrayStride);

    VkClusterAccelerationStructureCommandsInfoNV commandsInfo = {
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};

    // Compact
    commandsInfo.input = inputInfo;
    commandsInfo.dstImplicitData =
        dstImplicitData ? device->GetDeviceAddress(dstImplicitData->buffer) : 0;
    commandsInfo.scratchData =
        scratchBuffer ? device->GetDeviceAddress(scratchBuffer->buffer) : 0;

    commandsInfo.dstAddressesArray.deviceAddress =
        dstAddresses ? device->GetDeviceAddress(dstAddresses->buffer) +
                           sizeof(VkDeviceAddress) * dstClasOffset
                     : 0;
    commandsInfo.dstAddressesArray.size   = dstAddresses ? dstAddresses->size : 0;
    commandsInfo.dstAddressesArray.stride = sizeof(VkDeviceAddress);

    commandsInfo.dstSizesArray.deviceAddress =
        dstSizes ? device->GetDeviceAddress(dstSizes->buffer) + sizeof(u32) * dstClasOffset
                 : 0;
    commandsInfo.dstSizesArray.size   = dstSizes ? dstSizes->size : 0;
    commandsInfo.dstSizesArray.stride = dstSizes ? sizeof(u32) : 0;

    commandsInfo.srcInfosArray.deviceAddress = device->GetDeviceAddress(srcInfosArray->buffer);
    commandsInfo.srcInfosArray.size          = srcInfosArray->size;
    commandsInfo.srcInfosArray.stride        = srcInfosArrayStride;

    commandsInfo.srcInfosCount =
        device->GetDeviceAddress(srcInfosCount->buffer) + srcInfosOffset;
    commandsInfo.addressResolutionFlags = {};

    vkCmdBuildClusterAccelerationStructureIndirectNV(buffer, &commandsInfo);
}

void CommandBuffer::CLASIndirect(CLASOpInput opInput, CLASOpMode opMode, CLASOpType opType,
                                 u64 dstImplicitData, u64 scratchBuffer, u64 dstAddresses,
                                 u32 dstAddressesSize, u64 dstSizes, u32 dstSizesSize,
                                 u64 srcInfosArray, u32 srcInfosArraySize, u64 srcInfosCount)
{

    VkDeviceSize srcInfosArrayStride;
    VkOpInput vkOpInput = {};
    VkClusterAccelerationStructureOpModeNV vkOpMode;

    switch (opMode)
    {
        case CLASOpMode::ImplicitDestinations:
            vkOpMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
            break;
        case CLASOpMode::ExplicitDestinations:
            vkOpMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
            break;
        case CLASOpMode::ComputeSizes:
            vkOpMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
            break;
    }

    VkClusterAccelerationStructureInputInfoNV inputInfo = {
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputInfo.opMode = vkOpMode;
    inputInfo.flags  = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

    device->ConvertCLASIndirectInfo(opInput, opType, inputInfo, vkOpInput,
                                    srcInfosArrayStride);

    VkClusterAccelerationStructureCommandsInfoNV commandsInfo = {
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};

    // Compact
    commandsInfo.input           = inputInfo;
    commandsInfo.dstImplicitData = dstImplicitData;
    commandsInfo.scratchData     = scratchBuffer;

    commandsInfo.dstAddressesArray.deviceAddress = dstAddresses;
    commandsInfo.dstAddressesArray.size          = dstAddressesSize;
    commandsInfo.dstAddressesArray.stride        = sizeof(VkDeviceAddress);

    commandsInfo.dstSizesArray.deviceAddress = dstSizes;
    commandsInfo.dstSizesArray.size          = dstSizesSize;
    commandsInfo.dstSizesArray.stride        = sizeof(u32);

    commandsInfo.srcInfosArray.deviceAddress = srcInfosArray;
    commandsInfo.srcInfosArray.size          = srcInfosArraySize;
    commandsInfo.srcInfosArray.stride        = srcInfosArrayStride;

    commandsInfo.srcInfosCount          = srcInfosCount;
    commandsInfo.addressResolutionFlags = {};

    vkCmdBuildClusterAccelerationStructureIndirectNV(buffer, &commandsInfo);
}

void CommandBuffer::ComputeCLASSizes(GPUBuffer *srcInfosArray, GPUBuffer *scratchBuffer,
                                     GPUBuffer *dstSizes, GPUBuffer *srcInfosCount,
                                     u32 srcInfosOffset, u32 dstClasOffset,
                                     u32 maxNumTriangles, u32 maxNumVertices,
                                     u32 maxNumClusters)
{
    CLASOpInput opInput;
    opInput.triangleClusters.maxNumTriangles     = maxNumTriangles;
    opInput.triangleClusters.maxNumVertices      = maxNumVertices;
    opInput.triangleClusters.maxClusterTriangles = MAX_CLUSTER_TRIANGLES;
    opInput.triangleClusters.maxClusterVertices  = MAX_CLUSTER_TRIANGLE_VERTICES;

    opInput.maxAccelerationStructureCount = maxNumClusters;

    CLASIndirect(opInput, CLASOpMode::ComputeSizes, CLASOpType::CLAS, 0, scratchBuffer, 0,
                 dstSizes, srcInfosArray, srcInfosCount, srcInfosOffset, dstClasOffset);
}

void CommandBuffer::ComputeCLASSizes(u64 srcInfosArray, u32 srcInfosArraySize,
                                     u64 scratchBuffer, u64 dstSizes, u32 dstSizesSize,
                                     u64 srcInfosCount, u32 maxNumTriangles,
                                     u32 maxNumVertices, u32 maxNumClusters)
{
    CLASOpInput opInput;
    opInput.triangleClusters.maxNumTriangles     = maxNumTriangles;
    opInput.triangleClusters.maxNumVertices      = maxNumVertices;
    opInput.triangleClusters.maxClusterTriangles = MAX_CLUSTER_TRIANGLES;
    opInput.triangleClusters.maxClusterVertices  = MAX_CLUSTER_TRIANGLE_VERTICES;

    opInput.maxAccelerationStructureCount = maxNumClusters;

    CLASIndirect(opInput, CLASOpMode::ComputeSizes, CLASOpType::CLAS, 0, scratchBuffer, 0, 0,
                 dstSizes, dstSizesSize, srcInfosArray, srcInfosArraySize, srcInfosCount);
}

void CommandBuffer::ComputeBLASSizes(GPUBuffer *srcInfosArray, GPUBuffer *scratchBuffer,
                                     GPUBuffer *dstSizes, GPUBuffer *srcInfosCount,
                                     u32 srcInfosOffset, u32 maxTotalClusterCount,
                                     u32 maxClusterCountPerAccelerationStructure,
                                     u32 maxAccelerationStructureCount, u32 blasOffset)
{
    CLASOpInput opInput;
    opInput.clusterBottomLevel.maxTotalClusterCount = maxTotalClusterCount;
    opInput.clusterBottomLevel.maxClusterCountPerAccelerationStructure =
        maxClusterCountPerAccelerationStructure;

    opInput.maxAccelerationStructureCount = maxAccelerationStructureCount;

    CLASIndirect(opInput, CLASOpMode::ComputeSizes, CLASOpType::BLAS, 0, scratchBuffer, 0,
                 dstSizes, srcInfosArray, srcInfosCount, srcInfosOffset, blasOffset);
}

void CommandBuffer::ComputeCLASTemplateSizes(GPUBuffer *srcInfosArray,
                                             GPUBuffer *scratchBuffer, GPUBuffer *dstSizes,
                                             GPUBuffer *srcInfosCount, u32 srcInfosOffset,
                                             u32 maxNumTriangles, u32 maxNumVertices,
                                             u32 maxClusterTriangles, u32 maxClusterVertices,
                                             u32 maxNumClusters, u32 dstAddressesOffset)
{
    CLASOpInput opInput;
    opInput.triangleClusters.maxNumTriangles     = maxNumTriangles;
    opInput.triangleClusters.maxNumVertices      = maxNumVertices;
    opInput.triangleClusters.maxClusterTriangles = maxClusterTriangles;
    opInput.triangleClusters.maxClusterVertices  = maxClusterVertices;

    CLASIndirect(opInput, CLASOpMode::ComputeSizes,
                 CLASOpType::CLAS | CLASOpType::InstantiateTemplate, 0, scratchBuffer, 0,
                 dstSizes, srcInfosArray, srcInfosCount, srcInfosOffset, dstAddressesOffset);
}

void CommandBuffer::BuildCLAS(CLASOpMode opMode, GPUBuffer *dstImplicitData,
                              GPUBuffer *scratchBuffer, GPUBuffer *triangleClusterInfo,
                              GPUBuffer *dstAddresses, GPUBuffer *dstSizes,
                              GPUBuffer *srcInfosCount, u32 srcInfosOffset, int maxNumClusters,
                              u32 maxNumTriangles, u32 maxNumVertices, u32 dstClasOffset)
{
    CLASOpInput opInput;
    opInput.triangleClusters.maxNumTriangles     = maxNumTriangles;
    opInput.triangleClusters.maxNumVertices      = maxNumVertices;
    opInput.triangleClusters.maxClusterTriangles = MAX_CLUSTER_TRIANGLES;
    opInput.triangleClusters.maxClusterVertices  = MAX_CLUSTER_TRIANGLE_VERTICES;

    opInput.maxAccelerationStructureCount = maxNumClusters;

    CLASIndirect(opInput, opMode, CLASOpType::CLAS, dstImplicitData, scratchBuffer,
                 dstAddresses, dstSizes, triangleClusterInfo, srcInfosCount, srcInfosOffset,
                 dstClasOffset);
}

void CommandBuffer::BuildCLAS(CLASOpMode opMode, u64 dstImplicitData, u64 scratchBuffer,
                              u64 srcInfosArray, u32 srcInfosArraySize, u64 dstAddressesArray,
                              u32 dstAddressesSize, u64 dstSizes, u32 dstSizesSize,
                              u64 srcInfosCount, u32 maxNumTriangles, u32 maxNumVertices,
                              u32 maxNumClusters)
{
    CLASOpInput opInput;
    opInput.triangleClusters.maxNumTriangles     = maxNumTriangles;
    opInput.triangleClusters.maxNumVertices      = maxNumVertices;
    opInput.triangleClusters.maxClusterTriangles = MAX_CLUSTER_TRIANGLES;
    opInput.triangleClusters.maxClusterVertices  = MAX_CLUSTER_TRIANGLE_VERTICES;

    opInput.maxAccelerationStructureCount = maxNumClusters;

    CLASIndirect(opInput, opMode, CLASOpType::CLAS, dstImplicitData, scratchBuffer,
                 dstAddressesArray, dstAddressesSize, dstSizes, dstSizesSize, srcInfosArray,
                 srcInfosArraySize, srcInfosCount);
}

void CommandBuffer::BuildClusterBLAS(CLASOpMode opMode, GPUBuffer *implicitBuffer,
                                     GPUBuffer *scratchBuffer, GPUBuffer *bottomLevelInfo,
                                     GPUBuffer *dstAddresses, GPUBuffer *dstSizes,
                                     GPUBuffer *srcInfosCount, u32 srcInfosOffset,
                                     u32 maxClusterCountPerAccelerationStructure,
                                     u32 maxTotalClusterCount,
                                     u32 maxAccelerationStructureCount, u32 blasOffset)
{
    CLASOpInput opInput;
    opInput.clusterBottomLevel.maxClusterCountPerAccelerationStructure =
        maxClusterCountPerAccelerationStructure;
    opInput.clusterBottomLevel.maxTotalClusterCount = maxTotalClusterCount;
    opInput.maxAccelerationStructureCount           = maxAccelerationStructureCount;

    CLASIndirect(opInput, opMode, CLASOpType::BLAS, implicitBuffer, scratchBuffer,
                 dstAddresses, dstSizes, bottomLevelInfo, srcInfosCount, srcInfosOffset,
                 blasOffset);
}

void CommandBuffer::MoveCLAS(CLASOpMode opMode, GPUBuffer *dstImplicitData,
                             GPUBuffer *scratchBuffer, GPUBuffer *dstAddresses,
                             GPUBuffer *dstSizes, GPUBuffer *srcInfosArray,
                             GPUBuffer *srcInfosCount, u32 srcInfosOffset, int maxNumClusters,
                             u64 maxMovedBytes, bool noMoveOverlap, u32 dstClasOffset)
{
    Assert((opMode == CLASOpMode::ImplicitDestinations && dstImplicitData) ||
           (opMode == CLASOpMode::ExplicitDestinations && !dstImplicitData));

    CLASOpInput opInput;
    opInput.moveObjects.maxNumClusters = maxNumClusters;
    opInput.moveObjects.maxMovedBytes  = maxMovedBytes;
    opInput.moveObjects.noMoveOverlap  = noMoveOverlap;

    opInput.maxAccelerationStructureCount = maxNumClusters;

    CLASIndirect(opInput, opMode, CLASOpType::Move | CLASOpType::CLAS, dstImplicitData,
                 scratchBuffer, dstAddresses, dstSizes, srcInfosArray, srcInfosCount,
                 srcInfosOffset, dstClasOffset);
}

void CommandBuffer::MoveCLAS(CLASOpMode opMode, u64 dstImplicitData, u64 scratchBuffer,
                             u64 srcInfosArray, u32 srcInfosArraySize, u64 dstAddressesArray,
                             u32 dstAddressesSize, u32 dstSizes, u32 dstSizesSize,
                             u64 srcInfosCount, int maxNumClusters, u64 maxMovedBytes,
                             bool noMoveOverlap)
{
    Assert((opMode == CLASOpMode::ImplicitDestinations && dstImplicitData) ||
           (opMode == CLASOpMode::ExplicitDestinations && !dstImplicitData));

    CLASOpInput opInput;
    opInput.moveObjects.maxNumClusters = maxNumClusters;
    opInput.moveObjects.maxMovedBytes  = maxMovedBytes;
    opInput.moveObjects.noMoveOverlap  = noMoveOverlap;

    opInput.maxAccelerationStructureCount = maxNumClusters;

    CLASIndirect(opInput, opMode, CLASOpType::Move | CLASOpType::CLAS, dstImplicitData,
                 scratchBuffer, dstAddressesArray, dstAddressesSize, dstSizes, dstSizesSize,
                 srcInfosArray, srcInfosArraySize, srcInfosCount);
}

void CommandBuffer::BuildCLASTemplates(CLASOpMode opMode, GPUBuffer *dstImplicitData,
                                       GPUBuffer *scratchBuffer, GPUBuffer *dstAddresses,
                                       GPUBuffer *dstSizes, GPUBuffer *srcInfosArray,
                                       GPUBuffer *srcInfosCount, u32 srcInfosOffset,
                                       u32 maxNumTriangles, u32 maxNumVertices,
                                       u32 maxClusterTriangles, u32 maxClusterVertices,
                                       u32 numTemplates)
{

    CLASOpInput opInput;
    opInput.triangleClusters.maxNumTriangles     = maxNumTriangles;
    opInput.triangleClusters.maxNumVertices      = maxNumVertices;
    opInput.triangleClusters.maxClusterTriangles = maxClusterTriangles;
    opInput.triangleClusters.maxClusterVertices  = maxClusterVertices;

    opInput.maxAccelerationStructureCount = numTemplates;

    CLASIndirect(opInput, opMode, CLASOpType::BuildTemplate | CLASOpType::CLAS,
                 dstImplicitData, scratchBuffer, dstAddresses, dstSizes, srcInfosArray,
                 srcInfosCount, srcInfosOffset);
}

void CommandBuffer::InstantiateCLASTemplate(CLASOpMode opMode, GPUBuffer *dstImplicitData,
                                            GPUBuffer *scratchBuffer, GPUBuffer *dstAddresses,
                                            GPUBuffer *dstSizes, GPUBuffer *srcInfosArray,
                                            GPUBuffer *srcInfosCount, u32 srcInfosOffset,
                                            u32 maxNumTriangles, u32 maxNumVertices,
                                            u32 maxClusterTriangles, u32 maxClusterVertices,
                                            u32 maxNumClusters, u32 dstClasOffset)
{
    CLASOpInput opInput;
    opInput.triangleClusters.maxNumTriangles     = maxNumTriangles;
    opInput.triangleClusters.maxNumVertices      = maxNumVertices;
    opInput.triangleClusters.maxClusterTriangles = maxClusterTriangles;
    opInput.triangleClusters.maxClusterVertices  = maxClusterVertices;

    opInput.maxAccelerationStructureCount = maxNumClusters;

    CLASIndirect(opInput, opMode, CLASOpType::InstantiateTemplate | CLASOpType::CLAS,
                 dstImplicitData, scratchBuffer, dstAddresses, dstSizes, srcInfosArray,
                 srcInfosCount, srcInfosOffset, dstClasOffset);
}

void CommandBuffer::BuildPTLAS(GPUBuffer *ptlasBuffer, GPUBuffer *scratchBuffer,
                               GPUBuffer *srcInfos, GPUBuffer *srcInfosCount,
                               u32 srcInfosOffset, u32 instanceCount,
                               u32 maxInstancesPerPartition, u32 partitionCount,
                               u32 maxInstanceInGlobalPartitionCount)
{

    VkPartitionedAccelerationStructureInstancesInputNV inputInfo = {
        VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT_NV};

    inputInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    inputInfo.instanceCount = instanceCount;
    inputInfo.maxInstancePerPartitionCount      = maxInstancesPerPartition;
    inputInfo.partitionCount                    = partitionCount;
    inputInfo.maxInstanceInGlobalPartitionCount = maxInstanceInGlobalPartitionCount;

    u64 ptlasAddress = device->GetDeviceAddress(ptlasBuffer->buffer);

    VkBuildPartitionedAccelerationStructureInfoNV info = {
        VK_STRUCTURE_TYPE_BUILD_PARTITIONED_ACCELERATION_STRUCTURE_INFO_NV};

    info.input                        = inputInfo;
    info.srcAccelerationStructureData = device->frameCount == 0 ? 0 : ptlasAddress;
    info.dstAccelerationStructureData = ptlasAddress;
    info.scratchData                  = device->GetDeviceAddress(scratchBuffer->buffer);
    info.srcInfos                     = device->GetDeviceAddress(srcInfos->buffer);
    info.srcInfosCount = device->GetDeviceAddress(srcInfosCount->buffer) + srcInfosOffset;

    vkCmdBuildPartitionedAccelerationStructuresNV(buffer, &info);
}

void CommandBuffer::BuildPTLAS(u64 ptlasBuffer, u64 scratchBuffer, u64 srcInfosBuffer,
                               u64 srcInfosCount, u32 instanceCount,
                               u32 maxInstancesPerPartition, u32 partitionCount,
                               u32 maxInstanceInGlobalPartitionCount)
{

    VkPartitionedAccelerationStructureInstancesInputNV inputInfo = {
        VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT_NV};

    inputInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    inputInfo.instanceCount = instanceCount;
    inputInfo.maxInstancePerPartitionCount      = maxInstancesPerPartition;
    inputInfo.partitionCount                    = partitionCount;
    inputInfo.maxInstanceInGlobalPartitionCount = maxInstanceInGlobalPartitionCount;

    VkBuildPartitionedAccelerationStructureInfoNV info = {
        VK_STRUCTURE_TYPE_BUILD_PARTITIONED_ACCELERATION_STRUCTURE_INFO_NV};

    info.input                        = inputInfo;
    info.srcAccelerationStructureData = device->frameCount == 0 ? 0 : ptlasBuffer;
    info.dstAccelerationStructureData = ptlasBuffer;
    info.scratchData                  = scratchBuffer;
    info.srcInfos                     = srcInfosBuffer;
    info.srcInfosCount                = srcInfosCount;

    vkCmdBuildPartitionedAccelerationStructuresNV(buffer, &info);
}

GPUAccelerationStructurePayload CommandBuffer::BuildTLAS(GPUBuffer *instanceData,
                                                         u32 numInstances)
{
    VkAccelerationStructureBuildRangeInfoKHR buildRange = {};
    buildRange.primitiveCount                           = numInstances;

    VkAccelerationStructureGeometryKHR geometry = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    auto &instances       = geometry.geometry.instances;
    instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.data.deviceAddress = device->GetDeviceAddress(instanceData->buffer);

    return BuildAS(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &geometry, 1, &buildRange,
                   &numInstances);
}

VkAccelerationStructureKHR CommandBuffer::BuildTLAS(GPUBuffer *accelBuffer,
                                                    GPUBuffer *scratchBuffer,
                                                    GPUBuffer *instanceData, u32 numInstances)
{
    VkAccelerationStructureBuildRangeInfoKHR buildRange = {};
    buildRange.primitiveCount                           = numInstances;

    VkAccelerationStructureGeometryKHR geometry = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    auto &instances       = geometry.geometry.instances;
    instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.data.deviceAddress = device->GetDeviceAddress(instanceData->buffer);

    return BuildAS(accelBuffer, scratchBuffer, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                   &geometry, 1, &buildRange, &numInstances);
}

void CommandBuffer::BuildTLAS(GPUBuffer *accelBuffer, GPUBuffer *scratchBuffer,
                              GPUBuffer *instanceData, StaticArray<AccelBuildInfo> &buildInfos,
                              u32 numTlas)
{
    ScratchArena scratch;

    VkAccelerationStructureGeometryKHR geometry = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    auto &instances       = geometry.geometry.instances;
    instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.data.deviceAddress = device->GetDeviceAddress(instanceData->buffer);

    auto *buildGeometryInfos = PushArrayNoZero(
        scratch.temp.arena, VkAccelerationStructureBuildGeometryInfoKHR, numTlas);
    VkAccelerationStructureBuildRangeInfoKHR **buildRangeInfos = PushArrayNoZero(
        scratch.temp.arena, VkAccelerationStructureBuildRangeInfoKHR *, buildInfos.Length());

    for (u32 i = 0; i < numTlas; i++)
    {
        AccelBuildInfo &tlasBuildInfo = buildInfos[i];
        buildRangeInfos[i] =
            PushStruct(scratch.temp.arena, VkAccelerationStructureBuildRangeInfoKHR);

        buildRangeInfos[i]->primitiveCount  = tlasBuildInfo.primitiveCount;
        buildRangeInfos[i]->primitiveOffset = tlasBuildInfo.primitiveOffset;

        auto &buildInfo = buildGeometryInfos[i];
        buildInfo       = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        buildInfo.type  = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.pGeometries               = &geometry;
        buildInfo.geometryCount             = 1;
        buildInfo.scratchData.deviceAddress = tlasBuildInfo.scratchDataDeviceAddress;
        buildInfo.dstAccelerationStructure  = tlasBuildInfo.as;
    }

    vkCmdBuildAccelerationStructuresKHR(buffer, numTlas, buildGeometryInfos, buildRangeInfos);
}

VkAccelerationStructureKHR CommandBuffer::BuildTLAS(GPUBuffer *accelBuffer,
                                                    GPUBuffer *scratchBuffer,
                                                    GPUBuffer *instanceData,
                                                    GPUBuffer *buildRangeBuffer,
                                                    u32 maxInstances)
{
    VkAccelerationStructureGeometryKHR geometry = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    auto &instances       = geometry.geometry.instances;
    instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.data.deviceAddress = device->GetDeviceAddress(instanceData->buffer);

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type  = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    buildInfo.pGeometries   = &geometry;
    buildInfo.geometryCount = 1;

    VkDeviceAddress scratchAddress = device->GetDeviceAddress(scratchBuffer->buffer);

    VkAccelerationStructureKHR as;

    VkAccelerationStructureCreateInfoKHR accelCreateInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};

    accelCreateInfo.buffer = accelBuffer->buffer;
    accelCreateInfo.size   = accelBuffer->size;
    accelCreateInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    vkCreateAccelerationStructureKHR(device->device, &accelCreateInfo, 0, &as);

    buildInfo.scratchData.deviceAddress = scratchAddress;
    buildInfo.dstAccelerationStructure  = as;

    VkAccelerationStructureDeviceAddressInfoKHR accelDeviceAddressInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    accelDeviceAddressInfo.accelerationStructure = as;

    VkDeviceAddress indirectDeviceAddress = device->GetDeviceAddress(buildRangeBuffer->buffer);
    u32 indirectStride                    = sizeof(VkDeviceAddress);

    u32 *ppMaxPrimitiveCounts[1] = {&maxInstances};

    vkCmdBuildAccelerationStructuresIndirectKHR(buffer, 1, &buildInfo, &indirectDeviceAddress,
                                                &indirectStride, ppMaxPrimitiveCounts);
    return as;
}

void Vulkan::CreateAccelerationStructures(
    StaticArray<AccelerationStructureCreate> &createInfos)
{
    for (int i = 0; i < createInfos.Length(); i++)
    {
        AccelerationStructureCreate &createInfo = createInfos[i];
        ErrorExit((createInfo.bufferOffset & 255) == 0,
                  "BLAS buffer offset must be a multiple of 256.\n");

        VkAccelerationStructureCreateInfoKHR accelCreateInfo = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};

        accelCreateInfo.buffer = createInfo.buffer->buffer;
        accelCreateInfo.offset = createInfo.bufferOffset;
        accelCreateInfo.size   = createInfo.accelSize;
        accelCreateInfo.type   = createInfo.type == AccelerationStructureType::Top
                                     ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
                                     : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

        vkCreateAccelerationStructureKHR(device, &accelCreateInfo, 0, &createInfo.as);

        VkAccelerationStructureDeviceAddressInfoKHR deviceAddressInfo = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        deviceAddressInfo.accelerationStructure = createInfo.as;

        createInfo.asDeviceAddress =
            vkGetAccelerationStructureDeviceAddressKHR(device, &deviceAddressInfo);
    }
}

void Vulkan::GetClusterBuildSizes(CLASOpInput opInput, CLASOpMode opMode, CLASOpType opType,
                                  u32 &scratchSize, u32 &updateScratchSize,
                                  u32 &accelerationStructureSize)
{
    VkDeviceSize srcInfosArrayStride;
    VkClusterAccelerationStructureOpModeNV vkOpMode;
    VkOpInput vkOpInput = {};

    switch (opMode)
    {
        case CLASOpMode::ImplicitDestinations:
            vkOpMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
            break;
        case CLASOpMode::ExplicitDestinations:
            vkOpMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
            break;
        case CLASOpMode::ComputeSizes:
            vkOpMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
            break;
    }

    VkClusterAccelerationStructureInputInfoNV inputInfo = {
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputInfo.opMode = vkOpMode;
    inputInfo.flags  = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

    ConvertCLASIndirectInfo(opInput, opType, inputInfo, vkOpInput, srcInfosArrayStride);

    VkAccelerationStructureBuildSizesInfoKHR buildSizesInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(device, &inputInfo, &buildSizesInfo);

    scratchSize               = buildSizesInfo.buildScratchSize;
    updateScratchSize         = buildSizesInfo.updateScratchSize;
    accelerationStructureSize = buildSizesInfo.accelerationStructureSize;
}

void Vulkan::GetCLASBuildSizes(CLASOpMode opMode, int maxNumClusters, u32 maxNumTriangles,
                               u32 maxNumVertices, u32 maxClusterTriangles,
                               u32 maxClusterVertices, u32 &scratchSize,
                               u32 &accelerationStructureSize)

{

    CLASOpInput opInput;
    opInput.triangleClusters.maxNumTriangles     = maxNumTriangles;
    opInput.triangleClusters.maxNumVertices      = maxNumVertices;
    opInput.triangleClusters.maxClusterTriangles = maxClusterTriangles;
    opInput.triangleClusters.maxClusterVertices  = maxClusterVertices;
    opInput.maxAccelerationStructureCount        = maxNumClusters;

    u32 updateScratchSize;
    GetClusterBuildSizes(opInput, opMode, CLASOpType::CLAS, scratchSize, updateScratchSize,
                         accelerationStructureSize);
}

void Vulkan::GetCLASTemplateBuildSizes(CLASOpMode opMode, int maxNumClusters,
                                       u32 maxNumTriangles, u32 maxNumVertices,
                                       u32 maxClusterTriangles, u32 maxClusterVertices,
                                       u32 &scratchSize, u32 &accelerationStructureSize)
{
    CLASOpInput opInput;
    opInput.triangleClusters.maxNumTriangles     = maxNumTriangles;
    opInput.triangleClusters.maxNumVertices      = maxNumVertices;
    opInput.triangleClusters.maxClusterTriangles = maxClusterTriangles;
    opInput.triangleClusters.maxClusterVertices  = maxClusterVertices;
    opInput.maxAccelerationStructureCount        = maxNumClusters;

    u32 updateScratchSize;
    GetClusterBuildSizes(opInput, opMode, CLASOpType::CLAS | CLASOpType::BuildTemplate,
                         scratchSize, updateScratchSize, accelerationStructureSize);
}

void Vulkan::GetCLASTemplateInstantiateSizes(CLASOpMode opMode, int maxNumClusters,
                                             u32 maxNumTriangles, u32 maxNumVertices,
                                             u32 maxClusterTriangles, u32 maxClusterVertices,
                                             u32 &scratchSize, u32 &accelerationStructureSize)
{
    CLASOpInput opInput;
    opInput.triangleClusters.maxNumTriangles     = maxNumTriangles;
    opInput.triangleClusters.maxNumVertices      = maxNumVertices;
    opInput.triangleClusters.maxClusterTriangles = maxClusterTriangles;
    opInput.triangleClusters.maxClusterVertices  = maxClusterVertices;
    opInput.maxAccelerationStructureCount        = maxNumClusters;

    u32 updateScratchSize;
    GetClusterBuildSizes(opInput, opMode, CLASOpType::CLAS | CLASOpType::InstantiateTemplate,
                         scratchSize, updateScratchSize, accelerationStructureSize);
}

void Vulkan::GetClusterBLASBuildSizes(CLASOpMode opMode, u32 maxTotalClusterCount,
                                      u32 maxClusterCountPerAccelerationStructure,
                                      u32 maxAccelerationStructureCount, u32 &scratchSize,
                                      u32 &accelerationStructureSize)
{

    CLASOpInput opInput;
    opInput.clusterBottomLevel.maxClusterCountPerAccelerationStructure =
        maxClusterCountPerAccelerationStructure;
    opInput.clusterBottomLevel.maxTotalClusterCount = maxTotalClusterCount;
    opInput.maxAccelerationStructureCount           = maxAccelerationStructureCount;
    u32 updateScratchSize;
    GetClusterBuildSizes(opInput, opMode, CLASOpType::BLAS, scratchSize, updateScratchSize,
                         accelerationStructureSize);
}

void Vulkan::GetMoveBuildSizes(CLASOpMode opMode, int maxNumClusters, u64 maxMovedBytes,
                               bool noMoveOverlap, u32 &scratchSize,
                               u32 &accelerationStructureSize)
{
    CLASOpInput opInput;
    opInput.moveObjects.maxNumClusters = maxNumClusters;
    opInput.moveObjects.maxMovedBytes  = maxMovedBytes;
    opInput.moveObjects.noMoveOverlap  = noMoveOverlap;

    opInput.maxAccelerationStructureCount = maxNumClusters;

    u32 buildSize;
    GetClusterBuildSizes(opInput, opMode, CLASOpType::Move | CLASOpType::CLAS, buildSize,
                         scratchSize, accelerationStructureSize);
}

void Vulkan::GetPTLASBuildSizes(u32 instanceCount, u32 maxInstancesPerPartition,
                                u32 partitionCount, u32 maxInstanceInGlobalPartitionCount,
                                u32 &scratchSize, u32 &accelSize)
{
    VkPartitionedAccelerationStructureInstancesInputNV inputInfo = {
        VK_STRUCTURE_TYPE_PARTITIONED_ACCELERATION_STRUCTURE_INSTANCES_INPUT_NV};

    inputInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    inputInfo.instanceCount = instanceCount;
    inputInfo.maxInstancePerPartitionCount      = maxInstancesPerPartition;
    inputInfo.partitionCount                    = partitionCount;
    inputInfo.maxInstanceInGlobalPartitionCount = maxInstanceInGlobalPartitionCount;

    VkAccelerationStructureBuildSizesInfoKHR buildSizes = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};

    vkGetPartitionedAccelerationStructuresBuildSizesNV(device, &inputInfo, &buildSizes);
    scratchSize = buildSizes.buildScratchSize;
    accelSize   = buildSizes.accelerationStructureSize;
}

VkAccelerationStructureKHR Vulkan::CreatePTLAS(GPUBuffer *tlasBuffer)
{
    VkAccelerationStructureCreateInfoKHR createInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};

    createInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    createInfo.buffer = tlasBuffer->buffer;
    createInfo.size   = tlasBuffer->size;

    VkAccelerationStructureKHR accel;
    vkCreateAccelerationStructureKHR(device, &createInfo, 0, &accel);
    return accel;
}

StaticArray<AccelerationStructureSizes>
Vulkan::GetBLASBuildSizes(Arena *inArena, StaticArray<AccelBuildInfo> &blasBuildInfos)
{
    StaticArray<AccelerationStructureSizes> sizes(inArena, blasBuildInfos.Length());
    for (AccelBuildInfo &blasBuildInfo : blasBuildInfos)
    {
        VkAccelerationStructureGeometryKHR geometry = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;

        auto &aabbs  = geometry.geometry.aabbs;
        aabbs.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
        aabbs.stride = sizeof(VkAabbPositionsKHR);

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        buildInfo.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                          VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
        buildInfo.pGeometries   = &geometry;
        buildInfo.geometryCount = 1;

        u32 maxPrimitiveCount = blasBuildInfo.primitiveCount;

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};

        vkGetAccelerationStructureBuildSizesKHR(
            device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
            &maxPrimitiveCount, &sizeInfo);

        AccelerationStructureSizes accelSizes;
        accelSizes.scratchSize = sizeInfo.buildScratchSize;
        accelSizes.accelSize   = sizeInfo.accelerationStructureSize;

        sizes.Push(accelSizes);
    }
    return sizes;
}

StaticArray<AccelerationStructureSizes>
Vulkan::GetTLASBuildSizes(Arena *inArena, StaticArray<AccelBuildInfo> &tlasBuildInfos)
{
    StaticArray<AccelerationStructureSizes> sizes(inArena, tlasBuildInfos.Length());
    for (AccelBuildInfo &tlasBuildInfo : tlasBuildInfos)
    {
        VkAccelerationStructureGeometryKHR geometry = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;

        auto &instances = geometry.geometry.instances;
        instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.pGeometries   = &geometry;
        buildInfo.geometryCount = 1;

        u32 maxPrimitiveCount = tlasBuildInfo.primitiveCount;

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};

        vkGetAccelerationStructureBuildSizesKHR(
            device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
            &maxPrimitiveCount, &sizeInfo);

        AccelerationStructureSizes accelSizes;
        accelSizes.scratchSize = sizeInfo.buildScratchSize;
        accelSizes.accelSize   = sizeInfo.accelerationStructureSize;

        sizes.Push(accelSizes);
    }
    return sizes;
}

void Vulkan::GetBuildSizes(VkAccelerationStructureTypeKHR accelType,
                           VkAccelerationStructureGeometryKHR *geometries, int count,
                           VkAccelerationStructureBuildRangeInfoKHR *buildRanges,
                           u32 *maxPrimitiveCounts, u32 &buildScratchSize, u32 &accelSize)
{
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type  = accelType;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    buildInfo.pGeometries   = geometries;
    buildInfo.geometryCount = count;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};

    vkGetAccelerationStructureBuildSizesKHR(device,
                                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &buildInfo, maxPrimitiveCounts, &sizeInfo);

    buildScratchSize = sizeInfo.buildScratchSize;
    accelSize        = sizeInfo.accelerationStructureSize;
}

void Vulkan::GetTLASBuildSizes(u32 numInstances, u32 &buildScratchSize, u32 &accelSize)
{
    VkAccelerationStructureBuildRangeInfoKHR buildRange = {};
    buildRange.primitiveCount                           = numInstances;

    VkAccelerationStructureGeometryKHR geometry = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    auto &instances       = geometry.geometry.instances;
    instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;

    GetBuildSizes(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &geometry, 1, &buildRange,
                  &numInstances, buildScratchSize, accelSize);
}

QueryPool Vulkan::CreateQuery(QueryType type, int count)
{
    VkQueryPoolCreateInfo info = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    switch (type)
    {
        case QueryType_CompactSize:
        {
            info.queryType  = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
            info.queryCount = count;
        }
        break;
        default: Assert(0);
    }
    QueryPool result;
    VK_CHECK(vkCreateQueryPool(device, &info, 0, &result.queryPool));
    result.count = count;
    return result;
}

VkAccelerationStructureInstanceKHR Vulkan::GetVkInstance(const AffineSpace &transform,
                                                         GPUAccelerationStructure &as)
{
    VkAccelerationStructureInstanceKHR instanceAs;
    instanceAs.transform                              = ConvertMatrix(transform);
    instanceAs.instanceCustomIndex                    = 0;
    instanceAs.mask                                   = 0xff;
    instanceAs.instanceShaderBindingTableRecordOffset = 0;
    instanceAs.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instanceAs.accelerationStructureReference = as.address;
    return instanceAs;
}

QueryPool CommandBuffer::GetCompactionSizes(const GPUAccelerationStructurePayload *as)
{
    QueryPool pool = device->CreateQuery(QueryType_CompactSize, 1);
    vkCmdResetQueryPool(buffer, pool.queryPool, 0, 1);
    ScratchArena scratch;

    VkAccelerationStructureKHR vkAs = as->as.as;

    vkCmdWriteAccelerationStructuresPropertiesKHR(
        buffer, 1, &vkAs, VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
        pool.queryPool, 0);
    return pool;
}

GPUAccelerationStructure CommandBuffer::CompactAS(QueryPool &pool,
                                                  const GPUAccelerationStructurePayload *as)
{
    VkDeviceSize compactedSize = 0;
    vkGetQueryPoolResults(device->device, pool.queryPool, 0, 1, sizeof(VkDeviceSize),
                          &compactedSize, 0,
                          VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT);

    Assert(compactedSize != 0);
    GPUBuffer newBuffer = device->CreateBuffer(
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, compactedSize);
    VkAccelerationStructureKHR newAs;

    VkAccelerationStructureCreateInfoKHR accelCreateInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};

    accelCreateInfo.buffer = newBuffer.buffer;
    accelCreateInfo.size   = compactedSize;
    accelCreateInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    vkCreateAccelerationStructureKHR(device->device, &accelCreateInfo, 0, &newAs);

    VkCopyAccelerationStructureInfoKHR copyInfo = {
        VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
    copyInfo.src  = as->as.as;
    copyInfo.dst  = newAs;
    copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;

    vkCmdCopyAccelerationStructureKHR(buffer, &copyInfo);

    VkAccelerationStructureDeviceAddressInfoKHR accelDeviceAddressInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    accelDeviceAddressInfo.accelerationStructure = newAs;

    GPUAccelerationStructure result;
    result.as     = newAs;
    result.buffer = newBuffer;
    result.address =
        vkGetAccelerationStructureDeviceAddressKHR(device->device, &accelDeviceAddressInfo);
    return result;
}

void CommandBuffer::ClearBuffer(GPUBuffer *b, u32 val, uint64_t dstOffset, uint64_t dstSize)
{
    vkCmdFillBuffer(buffer, b->buffer, dstOffset, dstSize, val);
    b->lastStage  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    b->lastAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;
}

void CommandBuffer::ClearImage(GPUImage *image, u32 value, u32 baseMip, u32 numMips,
                               u32 baseLayer, u32 numLayers)
{
    VkClearColorValue colorValue = {};
    colorValue.uint32[0]         = value;
    VkImageSubresourceRange range;
    range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel   = baseMip;
    range.levelCount     = numMips;
    range.baseArrayLayer = baseLayer;
    range.layerCount     = numLayers;
    vkCmdClearColorImage(buffer, image->image, image->lastLayout, &colorValue, 1, &range);
}

void CommandBuffer::ClearImage(GPUImage *image, const Vec4f &value, u32 baseMip, u32 numMips,
                               u32 baseLayer, u32 numLayers)
{
    VkClearColorValue colorValue = {};
    colorValue.float32[0]        = value.x;
    colorValue.float32[1]        = value.y;
    colorValue.float32[2]        = value.z;
    colorValue.float32[3]        = value.w;

    VkImageSubresourceRange range;
    range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel   = baseMip;
    range.levelCount     = numMips;
    range.baseArrayLayer = baseLayer;
    range.layerCount     = numLayers;
    vkCmdClearColorImage(buffer, image->image, image->lastLayout, &colorValue, 1, &range);
}

void Vulkan::CreateQueryPool(QueryPool *queryPool, QueryType type, u32 queryCount)
{
    queryPool->type                  = type;
    queryPool->count                 = queryCount;
    VkQueryPoolCreateInfo createInfo = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    createInfo.queryCount            = queryCount;
    switch (type)
    {
        case QueryType_PipelineStatistics:
        {
            createInfo.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
            createInfo.pipelineStatistics =
                VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
                VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT;
        }
        break;
        case QueryType_Timestamp: createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP; break;
        case QueryType_Occlusion: createInfo.queryType = VK_QUERY_TYPE_OCCLUSION; break;
        default: Assert(0);
    }

    VkQueryPool vkQueryPool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateQueryPool(device, &createInfo, 0, &vkQueryPool));

    queryPool->queryPool = vkQueryPool;
}

void CommandBuffer::BeginQuery(QueryPool *queryPool, u32 queryIndex)
{
    VkQueryPool queryPoolVulkan = queryPool->queryPool;
    switch (queryPool->type)
    {
        case QueryType_PipelineStatistics:
            vkCmdBeginQuery(buffer, queryPoolVulkan, queryIndex, 0);
            break;
        case QueryType_Timestamp: break;
        case QueryType_Occlusion:
            vkCmdBeginQuery(buffer, queryPoolVulkan, queryIndex, VK_QUERY_CONTROL_PRECISE_BIT);
            break;
    }
}

void CommandBuffer::EndQuery(QueryPool *queryPool, u32 queryIndex)
{
    VkQueryPool queryPoolVulkan = queryPool->queryPool;
    switch (queryPool->type)
    {
        case QueryType_Occlusion:
        case QueryType_PipelineStatistics:
            vkCmdEndQuery(buffer, queryPoolVulkan, queryIndex);
            break;
        case QueryType_Timestamp:
            vkCmdWriteTimestamp2(buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queryPoolVulkan,
                                 queryIndex);
            break;
    }
}

void CommandBuffer::ResolveQuery(QueryPool *queryPool, GPUBuffer *gpuBuffer, u32 queryIndex,
                                 u32 count, u32 destOffset)
{
    VkQueryResultFlags flags = VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT;
    vkCmdCopyQueryPoolResults(buffer, queryPool->queryPool, queryIndex, count,
                              gpuBuffer->buffer, destOffset, sizeof(u64), flags);
}

void CommandBuffer::ResetQuery(QueryPool *queryPool, u32 index, u32 count)
{
    vkCmdResetQueryPool(buffer, queryPool->queryPool, index, count);
}

#ifdef USE_DLSS
DLSSTargets Vulkan::InitializeDLSSTargets(GPUImage *inColor, GPUImage *inDiffuseAlbedo,
                                          GPUImage *inSpecularAlbedo,
                                          GPUImage *inNormalRoughness, GPUImage *inMvec,
                                          GPUImage *inDepth, GPUImage *inSpecularHitDistance,
                                          GPUImage *outColor)
{
    DLSSTargets targets;
    targets.diffuseAlbedo              = {sl::ResourceType::eTex2d, inDiffuseAlbedo->image,
                                          inDiffuseAlbedo->allocation->GetMemory(),
                                          inDiffuseAlbedo->imageView,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    targets.diffuseAlbedo.width        = inDiffuseAlbedo->desc.width;
    targets.diffuseAlbedo.height       = inDiffuseAlbedo->desc.height;
    targets.diffuseAlbedo.nativeFormat = inDiffuseAlbedo->desc.format;
    targets.diffuseAlbedo.mipLevels    = inDiffuseAlbedo->desc.numMips;
    targets.diffuseAlbedo.arrayLayers  = inDiffuseAlbedo->desc.numLayers;
    targets.diffuseAlbedo.usage        = inDiffuseAlbedo->desc.imageUsage;
    targets.diffuseAlbedo.flags        = 0;

    targets.specularAlbedo              = {sl::ResourceType::eTex2d, inSpecularAlbedo->image,
                                           inSpecularAlbedo->allocation->GetMemory(),
                                           inSpecularAlbedo->imageView,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    targets.specularAlbedo.width        = inSpecularAlbedo->desc.width;
    targets.specularAlbedo.height       = inSpecularAlbedo->desc.height;
    targets.specularAlbedo.nativeFormat = inSpecularAlbedo->desc.format;
    targets.specularAlbedo.mipLevels    = inSpecularAlbedo->desc.numMips;
    targets.specularAlbedo.arrayLayers  = inSpecularAlbedo->desc.numLayers;
    targets.specularAlbedo.usage        = inSpecularAlbedo->desc.imageUsage;
    targets.specularAlbedo.flags        = 0;

    targets.normalRoughness              = {sl::ResourceType::eTex2d, inNormalRoughness->image,
                                            inNormalRoughness->allocation->GetMemory(),
                                            inNormalRoughness->imageView,
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    targets.normalRoughness.width        = inNormalRoughness->desc.width;
    targets.normalRoughness.height       = inNormalRoughness->desc.height;
    targets.normalRoughness.nativeFormat = inNormalRoughness->desc.format;
    targets.normalRoughness.mipLevels    = inNormalRoughness->desc.numMips;
    targets.normalRoughness.arrayLayers  = inNormalRoughness->desc.numLayers;
    targets.normalRoughness.usage        = inNormalRoughness->desc.imageUsage;
    targets.normalRoughness.flags        = 0;

    targets.colorIn              = {sl::ResourceType::eTex2d, inColor->image,
                                    inColor->allocation->GetMemory(), inColor->imageView,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    targets.colorIn.width        = inColor->desc.width;
    targets.colorIn.height       = inColor->desc.height;
    targets.colorIn.nativeFormat = inColor->desc.format;
    targets.colorIn.mipLevels    = inColor->desc.numMips;
    targets.colorIn.arrayLayers  = inColor->desc.numLayers;
    targets.colorIn.usage        = inColor->desc.imageUsage;
    targets.colorIn.flags        = 0;
    targets.mvec = {sl::ResourceType::eTex2d, inMvec->image, inMvec->allocation->GetMemory(),
                    inMvec->imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    targets.mvec.width        = inMvec->desc.width;
    targets.mvec.height       = inMvec->desc.height;
    targets.mvec.nativeFormat = inMvec->desc.format;
    targets.mvec.mipLevels    = inMvec->desc.numMips;
    targets.mvec.arrayLayers  = inMvec->desc.numLayers;
    targets.mvec.usage        = inMvec->desc.imageUsage;
    targets.mvec.flags        = 0;

    targets.depth              = {sl::ResourceType::eTex2d, inDepth->image,
                                  inDepth->allocation->GetMemory(), inDepth->imageView,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    targets.depth.width        = inDepth->desc.width;
    targets.depth.height       = inDepth->desc.height;
    targets.depth.nativeFormat = inDepth->desc.format;
    targets.depth.mipLevels    = inDepth->desc.numMips;
    targets.depth.arrayLayers  = inDepth->desc.numLayers;
    targets.depth.usage        = inDepth->desc.imageUsage;
    targets.depth.flags        = 0;

    targets.specularHitDistance = {sl::ResourceType::eTex2d, inSpecularHitDistance->image,
                                   inSpecularHitDistance->allocation->GetMemory(),
                                   inSpecularHitDistance->imageView,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    targets.specularHitDistance.width        = inSpecularHitDistance->desc.width;
    targets.specularHitDistance.height       = inSpecularHitDistance->desc.height;
    targets.specularHitDistance.nativeFormat = inSpecularHitDistance->desc.format;
    targets.specularHitDistance.mipLevels    = inSpecularHitDistance->desc.numMips;
    targets.specularHitDistance.arrayLayers  = inSpecularHitDistance->desc.numLayers;
    targets.specularHitDistance.usage        = inSpecularHitDistance->desc.imageUsage;
    targets.specularHitDistance.flags        = 0;

    targets.colorOut              = {sl::ResourceType::eTex2d, outColor->image,
                                     outColor->allocation->GetMemory(), outColor->imageView,
                                     VK_IMAGE_LAYOUT_GENERAL};
    targets.colorOut.width        = outColor->desc.width;
    targets.colorOut.height       = outColor->desc.height;
    targets.colorOut.nativeFormat = outColor->desc.format;
    targets.colorOut.mipLevels    = outColor->desc.numMips;
    targets.colorOut.arrayLayers  = outColor->desc.numLayers;
    targets.colorOut.usage        = outColor->desc.imageUsage;
    targets.colorOut.flags        = 0;
    return targets;
}

void CommandBuffer::DLSS(DLSSTargets &targets, AffineSpace &worldToCameraView,
                         AffineSpace &cameraViewToWorld, Mat4 &clipFromCamera,
                         Mat4 &cameraFromClip, Mat4 &clipToPrevClip, Mat4 &prevClipToClip,
                         Vec3f &cameraP, Vec3f &cameraUp, Vec3f &cameraFwd, Vec3f &cameraRight,
                         f32 fov, f32 aspectRatio, const Vec2f &jitter)
{
    static sl::ViewportHandle viewportHandle = {};

    sl::Extent inExtent{0, 0, targets.colorIn.width, targets.colorIn.height};
    sl::Extent outExtent{0, 0, targets.colorOut.width, targets.colorOut.height};
    sl::ResourceTag diffuseAlbedoTag   = {&targets.diffuseAlbedo, sl::kBufferTypeAlbedo,
                                          sl::ResourceLifecycle::eValidUntilPresent, &inExtent};
    sl::ResourceTag specularAlbedoTag  = {&targets.specularAlbedo,
                                          sl::kBufferTypeSpecularAlbedo,
                                          sl::ResourceLifecycle::eValidUntilPresent, &inExtent};
    sl::ResourceTag normalRoughnessTag = {
        &targets.normalRoughness, sl::kBufferTypeNormalRoughness,
        sl::ResourceLifecycle::eValidUntilPresent, &inExtent};
    sl::ResourceTag colorInTag = {&targets.colorIn, sl::kBufferTypeScalingInputColor,
                                  sl::ResourceLifecycle::eOnlyValidNow, &inExtent};
    sl::ResourceTag mvecTag    = {&targets.mvec, sl::kBufferTypeMotionVectors,
                                  sl::ResourceLifecycle::eValidUntilPresent, &inExtent};
    sl::ResourceTag depthTag   = {&targets.depth, sl::kBufferTypeDepth,
                                  sl::ResourceLifecycle::eValidUntilPresent, &inExtent};
    sl::ResourceTag specularHitDistanceTag = {
        &targets.specularHitDistance, sl::kBufferTypeSpecularHitDistance,
        sl::ResourceLifecycle::eValidUntilPresent, &inExtent};
    sl::ResourceTag colorOutTag = {&targets.colorOut, sl::kBufferTypeScalingOutputColor,
                                   sl::ResourceLifecycle::eOnlyValidNow, &outExtent};
    sl::ResourceTag tags[]      = {
        diffuseAlbedoTag, specularAlbedoTag, normalRoughnessTag,     colorInTag,
        mvecTag,          depthTag,          specularHitDistanceTag, colorOutTag};

    sl::FrameToken *token = 0;
    u32 frameCount        = device->frameCount;
    sl::Result res        = slGetNewFrameToken_p(token, &frameCount);
    Assert(res == sl::Result::eOk);

    sl::DLSSDOptions options    = {};
    options.mode                = sl::DLSSMode::eMaxQuality;
    options.outputWidth         = 2560;
    options.outputHeight        = 1440;
    options.colorBuffersHDR     = sl::Boolean::eTrue;
    options.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::ePacked;
    for (int i = 0; i < 3; i++)
    {
        options.worldToCameraView.setRow(
            i, sl::float4(worldToCameraView[0][i], worldToCameraView[1][i],
                          worldToCameraView[2][i], worldToCameraView[3][i]));
        options.cameraViewToWorld.setRow(
            i, sl::float4(cameraViewToWorld[0][i], cameraViewToWorld[1][i],
                          cameraViewToWorld[2][i], cameraViewToWorld[3][i]));
    }
    options.worldToCameraView.setRow(3, sl::float4(0, 0, 0, 1));
    options.cameraViewToWorld.setRow(3, sl::float4(0, 0, 0, 1));
    res = slDLSSDSetOptions(viewportHandle, options);
    Assert(res == sl::Result::eOk);

    res = slSetTagForFrame_p(*token, viewportHandle, tags, ArrayLength(tags), buffer);
    Assert(res == sl::Result::eOk);

    auto ConvertToFloat4x4 = [&](Mat4 &transform) {
        sl::float4x4 result;
        for (int i = 0; i < 4; i++)
        {
            result.setRow(i, sl::float4(transform[0][i], transform[1][i], transform[2][i],
                                        transform[3][i]));
        }
        return result;
    };

    sl::Constants constants         = {};
    constants.mvecScale             = {1, 1};
    constants.cameraViewToClip      = ConvertToFloat4x4(cameraFromClip);
    constants.clipToCameraView      = ConvertToFloat4x4(clipFromCamera);
    constants.clipToPrevClip        = ConvertToFloat4x4(clipToPrevClip);
    constants.prevClipToClip        = ConvertToFloat4x4(prevClipToClip);
    constants.cameraPos             = sl::float3(cameraP.x, cameraP.y, cameraP.z);
    constants.cameraUp              = sl::float3(cameraUp.x, cameraUp.y, cameraUp.z);
    constants.cameraFwd             = sl::float3(cameraFwd.x, cameraFwd.y, cameraFwd.z);
    constants.cameraRight           = sl::float3(cameraRight.x, cameraRight.y, cameraRight.z);
    constants.cameraNear            = 1e-2f;
    constants.cameraFar             = pos_inf;
    constants.cameraFOV             = fov;
    constants.cameraAspectRatio     = aspectRatio;
    constants.cameraMotionIncluded  = sl::Boolean::eTrue;
    constants.depthInverted         = sl::Boolean::eTrue;
    constants.motionVectors3D       = sl::Boolean::eFalse;
    constants.motionVectorsJittered = sl::Boolean::eFalse;
    constants.reset = device->frameCount == 0 ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.jitterOffset = sl::float2(jitter.x, jitter.y);

    res = slSetConstants_p(constants, *token, viewportHandle);
    Assert(res == sl::Result::eOk);

    const sl::BaseStructure *inputs[] = {&viewportHandle};
    res =
        slEvaluateFeature_p(sl::kFeatureDLSS_RR, *token, inputs, ArrayLength(inputs), buffer);
    Assert(res == sl::Result::eOk);
}
#endif

u32 Vulkan::GetQueueFamily(QueueType queueType)
{
    Assert(families.size() > 0);
    u32 family = queueType == QueueType_Graphics
                     ? graphicsFamily
                     : (queueType == QueueType_Copy
                            ? copyFamily
                            : (queueType == QueueType_Compute ? computeFamily
                                                              : VK_QUEUE_FAMILY_IGNORED));
    return family;
}

bool Vulkan::BeginFrame(bool doubleBuffer)
{
    u32 threshold = doubleBuffer ? 2 : 1;

    bool success = true;
    if (frameCount >= threshold)
    {
        for (int i = 0; i < QueueType_Count; i++)
        {
            CommandQueue &queue = queues[i];
            if (queue.submissionID >= threshold)
            {
                u64 val         = queue.submissionID - doubleBuffer;
                VkResult result = vkWaitForFences(
                    device, 1, &queue.submitFence[GetCurrentBuffer()], true, 10e9);
                // VkSemaphoreWaitInfo waitInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
                // waitInfo.pValues             = &val;
                // waitInfo.semaphoreCount      = 1;
                // // waitInfo.pSemaphores         =
                // &queue.submitSemaphore[GetCurrentBuffer()]; waitInfo.pSemaphores =
                // &queue.submitSemaphore[GetCurrentBuffer()];
                // VkResult result = vkWaitSemaphores(device, &waitInfo, 10e9); //
                // UINT64_MAX);

                if (result != VK_SUCCESS)
                {
                    success = false;
                }
                vkResetFences(device, 1, &queue.submitFence[GetCurrentBuffer()]);
            }
        }
    }
    return success;
}
void Vulkan::EndFrame(int queueTypes)
{
    for (int i = 0; i < QueueType_Count; i++)
    {
        if ((queueTypes & (1 << i)) != 0)
        {
            CommandQueue &queue = queues[i];
            queue.submissionID++;
        }
    }
    frameCount++;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

bool Vulkan::Wait(Semaphore s, u64 waitVal)
{
    u64 val                      = s.signalValue;
    VkSemaphoreWaitInfo waitInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    waitInfo.pValues             = &val;
    waitInfo.semaphoreCount      = 1;
    waitInfo.pSemaphores         = &s.semaphore;
    VkResult result              = vkWaitSemaphores(device, &waitInfo, waitVal);

    return result == VK_SUCCESS;
}

u64 Vulkan::GetSemaphoreValue(Semaphore s)
{
    u64 value = 0;
    vkGetSemaphoreCounterValue(device, s.semaphore, &value);
    return value;
}

#ifdef WIN32
HANDLE Vulkan::GetWin32Handle(GPUImage *image)
{
    VkMemoryGetWin32HandleInfoKHR info = {VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    info.memory                        = image->allocation->GetMemory();
    info.handleType                    = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE result                      = 0;
    VkResult success                   = vkGetMemoryWin32HandleKHR(device, &info, &result);
    Assert(success == VK_SUCCESS);
    return result;
}

HANDLE Vulkan::GetWin32Handle(GPUBuffer *buffer)
{
    VkExternalMemoryHandleTypeFlagBits handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE result = 0;
    VkResult success =
        vmaGetMemoryWin32Handle2(allocator, buffer->allocation, handleType, nullptr, &result);

    Assert(success == VK_SUCCESS);
    return result;
}
#endif

void Vulkan::InitializeExternalMemoryPool()
{
    const VkExternalMemoryHandleTypeFlagsKHR handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;

    // Define an example buffer and allocation parameters.
    VkExternalMemoryBufferCreateInfoKHR externalMemBufCreateInfo = {
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR, nullptr, handleType};
    VkBufferCreateInfo exampleBufCreateInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    exampleBufCreateInfo.size               = 0x10000; // Doesn't matter here.
    exampleBufCreateInfo.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    exampleBufCreateInfo.pNext = &externalMemBufCreateInfo;

    VmaAllocationCreateInfo exampleAllocCreateInfo = {};
    exampleAllocCreateInfo.usage                   = VMA_MEMORY_USAGE_AUTO;

    // Find memory type index to use for the custom pool.
    uint32_t memTypeIndex;
    VK_CHECK(vmaFindMemoryTypeIndexForBufferInfo(allocator, &exampleBufCreateInfo,
                                                 &exampleAllocCreateInfo, &memTypeIndex));
    externalMemTypeIndex = memTypeIndex;

    // Create a custom pool.
    constexpr static VkExportMemoryAllocateInfoKHR exportMemAllocInfo = {
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR, nullptr, handleType};
    VmaPoolCreateInfo poolCreateInfo   = {};
    poolCreateInfo.memoryTypeIndex     = memTypeIndex;
    poolCreateInfo.pMemoryAllocateNext = (void *)&exportMemAllocInfo;

    VK_CHECK(vmaCreatePool(allocator, &poolCreateInfo, &externalPool));
}

#ifdef USE_DLSS
void Vulkan::GetDLSSTargetDimensions(u32 &width, u32 &height)
{
    // DLSS
    sl::DLSSDOptions dlssOptions = {};
    dlssOptions.mode             = sl::DLSSMode::eMaxQuality;
    dlssOptions.outputWidth      = 2560;
    dlssOptions.outputHeight     = 1440;

    sl::DLSSDOptimalSettings dlssSettings;
    sl::Result res = slDLSSDGetOptimalSettings(dlssOptions, dlssSettings);
    width          = dlssSettings.optimalRenderWidth;
    height         = dlssSettings.optimalRenderHeight;
}
#endif

} // namespace rt
