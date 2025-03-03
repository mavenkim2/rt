#include "vulkan.h"

namespace rt
{
Vulkan::Vulkan(ValidationMode validationMode, GPUDevicePreference preference)
{
    arena           = ArenaAlloc();
    const i32 major = 0;
    const i32 minor = 0;
    const i32 patch = 1;

    VK_CHECK(volkInitialize());

    // Create the application
    VkApplicationInfo appInfo  = {};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = "RT";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    appInfo.pEngineName        = "RT";
    appInfo.engineVersion      = VK_MAKE_API_VERSION(0, major, minor, patch);
    appInfo.apiVersion         = VK_API_VERSION_1_4;

    // Load available layers
    u32 layerCount = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&layerCount, 0));
    list<VkLayerProperties> availableLayers(layerCount);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()));

    // Load extension info
    u32 extensionCount = 0;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(0, &extensionCount, 0));
    list<VkExtensionProperties> extensionProperties(extensionCount);
    VK_CHECK(vkEnumerateInstanceExtensionProperties(0, &extensionCount,
                                                    extensionProperties.data()));

    list<const char *> instanceExtensions;
    list<const char *> instanceLayers;
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
        static const list<const char *> validationPriorityList[] = {
            // Preferred
            {"VK_LAYER_KHRONOS_validation"},
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
        Assert(volkGetInstanceVersion() >= VK_API_VERSION_1_4);
        VkInstanceCreateInfo instInfo    = {};
        instInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instInfo.pApplicationInfo        = &appInfo;
        instInfo.enabledLayerCount       = (u32)instanceLayers.size();
        instInfo.ppEnabledLayerNames     = instanceLayers.data();
        instInfo.enabledExtensionCount   = (u32)instanceExtensions.size();
        instInfo.ppEnabledExtensionNames = instanceExtensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugUtilsCreateInfo = {};

        debugUtilsCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        if (validationMode != ValidationMode::Disabled && debugUtils)
        {
            debugUtilsCreateInfo.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
            debugUtilsCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            if (validationMode == ValidationMode::Verbose)
            {
                debugUtilsCreateInfo.messageSeverity |=
                    (VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT);
            }

            debugUtilsCreateInfo.pfnUserCallback = DebugUtilsMessengerCallback;
            instInfo.pNext                       = &debugUtilsCreateInfo;
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

        list<VkPhysicalDevice> devices(deviceCount);
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()));

        list<const char *> deviceExtensions = {
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

            list<VkQueueFamilyProperties2> queueFamilyProps;
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
            if (props.properties.apiVersion < VK_API_VERSION_1_4) continue;

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
        features13.pNext     = &features14;

        void **featuresChain = &features14.pNext;
        *featuresChain       = 0;

        deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties11.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
        properties12.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
        properties13.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
        properties14.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES;
        deviceProperties.pNext = &properties11;
        properties11.pNext     = &properties12;
        properties12.pNext     = &properties13;
        properties13.pNext     = &properties14;
        void **propertiesChain = &properties14.pNext;

        u32 deviceExtCount = 0;
        VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, 0, &deviceExtCount, 0));
        list<VkExtensionProperties> availableDevExt(deviceExtCount);
        VK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, 0, &deviceExtCount,
                                                      availableDevExt.data()));

        auto checkAndAddExtension = [&](const char *extName, auto *prop = 0, auto *feat = 0) {
            for (auto &extension : availableDevExt)
            {
                if (strcmp(extension.extensionName, extName) == 0)
                {
                    if (prop)
                    {
                        *propertiesChain = prop;
                        propertiesChain  = &prop->pNext;
                    }
                    if (feat)
                    {
                        *featuresChain = &feat;
                        featuresChain  = &feat->pNext;
                    }
                    deviceExtensions.push_back(extName);
                    return true;
                }
            }
            return false;
        };

        if (checkExtension(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME))
        {
            deviceExtensions.push_back(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME);
        }

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

            checkAndAddExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

            clasPropertiesNV = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};
            clasFeaturesNV = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV};
            bool result =
                checkAndAddExtension(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                     &clasPropertiesNV, &clasFeaturesNV);
            ErrorExit(
                result,
                "Machine doesn't support VK_NV_cluster_acceleration_structure. Exiting\n");
        }

        vkGetPhysicalDeviceFeatures2(physicalDevice, &deviceFeatures);

        // Ensure core functionlity is supported
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
        queueFamilyProperties.resize(queueFamilyCount);
        for (u32 i = 0; i < queueFamilyCount; i++)
        {
            queueFamilyProperties[i].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        }
        vkGetPhysicalDeviceQueueFamilyProperties2(physicalDevice, &queueFamilyCount,
                                                  queueFamilyProperties.data());

        // Device exposes 1+ queue families, queue families have 1+ queues. Each family
        // supports a combination of the below:
        // 1. Graphics
        // 2. Compute
        // 3. Transfer
        // 4. Sparse Memory Management

        // Find queues in queue family
        for (u32 i = 0; i < queueFamilyProperties.size(); i++)
        {
            auto &queueFamily = queueFamilyProperties[i];
            if (queueFamily.queueFamilyProperties.queueCount > 0)
            {
                if (graphicsFamily == VK_QUEUE_FAMILY_IGNORED &&
                    queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                {
                    graphicsFamily = i;
                }
                if ((queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                    (copyFamily == VK_QUEUE_FAMILY_IGNORED ||
                     (!(queueFamily.queueFamilyProperties.queueFlags &
                        VK_QUEUE_GRAPHICS_BIT) &&
                      !(queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_COMPUTE_BIT))))

                {
                    copyFamily = i;
                }
                if ((queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                    (computeFamily == VK_QUEUE_FAMILY_IGNORED ||
                     !(queueFamily.queueFamilyProperties.queueFlags & VK_QUEUE_GRAPHICS_BIT)))

                {
                    computeFamily = i;
                }
            }
        }

        // Create the device queues
        list<VkDeviceQueueCreateInfo> queueCreateInfos;
        f32 queuePriority = 1.f;
        for (u32 i = 0; i < 3; i++)
        {
            u32 queueFamily = 0;
            if (i == 0)
            {
                queueFamily = graphicsFamily;
            }
            else if (i == 1)
            {
                if (graphicsFamily == computeFamily)
                {
                    continue;
                }
                queueFamily = computeFamily;
            }
            else if (i == 2)
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

            families.push_back(queueFamily);
        }

        VkDeviceCreateInfo createInfo      = {};
        createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount    = (u32)queueCreateInfos.size();
        createInfo.pQueueCreateInfos       = queueCreateInfos.data();
        createInfo.pEnabledFeatures        = 0;
        createInfo.pNext                   = &deviceFeatures;
        createInfo.enabledExtensionCount   = (u32)deviceExtensions.size();
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

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
    SetName(queues[QueueType_Copy].queue, "Transfer Queue");

    // TODO: unified memory access architectures
    memProperties       = {};
    memProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memProperties);

    VmaAllocatorCreateInfo allocCreateInfo = {};
    allocCreateInfo.physicalDevice         = physicalDevice;
    allocCreateInfo.device                 = device;
    allocCreateInfo.instance               = instance;
    allocCreateInfo.vulkanApiVersion       = VK_API_VERSION_1_4;
    // these are promoted to core, so this doesn't do anything
    allocCreateInfo.flags = VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;

#if VMA_DYNAMIC_VULKAN_FUNCTIONS
    VmaVulkanFunctions vulkanFunctions    = {};
    vulkanFunctions.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    allocCreateInfo.pVulkanFunctions      = &vulkanFunctions;
#else
#error
#endif

    VK_CHECK(vmaCreateAllocator(&allocCreateInfo, &allocator));

    // Set up dynamic pso
    dynamicStates = {
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_VIEWPORT,
    };

    // Set up frame fences
    for (u32 buffer = 0; buffer < cNumBuffers; buffer++)
    {
        for (u32 queue = 0; queue < QueueType_Count; queue++)
        {
            if (queues[queue].queue == VK_NULL_HANDLE)
            {
                continue;
            }
            VkFenceCreateInfo fenceInfo = {};
            fenceInfo.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VK_CHECK(vkCreateFence(device, &fenceInfo, 0, &frameFences[buffer][queue]));
        }
    }

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

        VK_CHECK(vkCreateDescriptorPool(device, &createInfo, 0, &pool));
    }

    // Bindless descriptor pools
    {
        for (DescriptorType type = (DescriptorType)0; type < DescriptorType_Count;
             type                = (DescriptorType)(type + 1))
        {
            VkDescriptorType descriptorType;
            switch (type)
            {
                case DescriptorType_SampledImage:
                    descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    break;
                case DescriptorType_UniformTexel:
                    descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
                    break;
                case DescriptorType_StorageBuffer:
                    descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    break;
                case DescriptorType_StorageTexelBuffer:
                    descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                    break;
                default: Assert(0);
            }

            BindlessDescriptorPool &bindlessDescriptorPool = bindlessDescriptorPools[type];
            VkDescriptorPoolSize poolSize                  = {};
            poolSize.type                                  = descriptorType;
            if (type == DescriptorType_StorageBuffer ||
                type == DescriptorType_StorageTexelBuffer)
            {
                poolSize.descriptorCount =
                    Min(10000,
                        deviceProperties.properties.limits.maxDescriptorSetStorageBuffers / 4);
            }
            else if (type == DescriptorType_SampledImage)
            {
                poolSize.descriptorCount =
                    Min(10000,
                        deviceProperties.properties.limits.maxDescriptorSetSampledImages / 4);
            }
            else if (type == DescriptorType_UniformTexel)
            {
                poolSize.descriptorCount =
                    Min(10000,
                        deviceProperties.properties.limits.maxDescriptorSetUniformBuffers / 4);
            }
            bindlessDescriptorPool.descriptorCount = poolSize.descriptorCount;

            VkDescriptorPoolCreateInfo createInfo = {};
            createInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
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
            TempArena temp = ScratchStart(0, 0);
            string typeName;
            switch (type)
            {
                case DescriptorType_SampledImage: typeName = "Sampled Image"; break;
                case DescriptorType_StorageBuffer: typeName = "Storage Buffer"; break;
                case DescriptorType_UniformTexel: typeName = "Uniform Texel Buffer"; break;
                case DescriptorType_StorageTexelBuffer:
                    typeName = "Storage Texel Buffer";
                    break;
            }
            string name =
                PushStr8F(temp.arena, "Bindless Descriptor Set Layout: %S", typeName);
            SetName(bindlessDescriptorPool.layout, (const char *)name.str);

            name = PushStr8F(temp.arena, "Bindless Descriptor Set: %S", typeName);
            SetName(bindlessDescriptorPool.set, (const char *)name.str);
            ScratchEnd(temp);
        }
    }

    // Init frame allocators
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

    // Default samplers
    {
        // Null sampler
        VkSamplerCreateInfo samplerCreate = {};
        samplerCreate.sType               = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

        VK_CHECK(vkCreateSampler(device, &samplerCreate, 0, &nullSampler));

        samplerCreate.anisotropyEnable        = VK_FALSE;
        samplerCreate.maxAnisotropy           = 0;
        samplerCreate.minLod                  = 0;
        samplerCreate.maxLod                  = FLT_MAX;
        samplerCreate.mipLodBias              = 0;
        samplerCreate.unnormalizedCoordinates = VK_FALSE;
        samplerCreate.compareEnable           = VK_FALSE;
        samplerCreate.compareOp               = VK_COMPARE_OP_NEVER;

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

        // sampler linear clamp
        samplerCreate.minFilter    = VK_FILTER_LINEAR;
        samplerCreate.magFilter    = VK_FILTER_LINEAR;
        samplerCreate.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
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

        // sampler nearest compare
        samplerCreate.compareEnable = VK_TRUE;
        samplerCreate.compareOp     = VK_COMPARE_OP_GREATER_OR_EQUAL;
        immutableSamplers.emplace_back();
        VK_CHECK(vkCreateSampler(device, &samplerCreate, 0, &immutableSamplers.back()));
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

        // Transitions
        TransferCommand cmd = Stage(0);

        VkImageMemoryBarrier2 imageBarrier       = {};
        imageBarrier.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        imageBarrier.image                       = nullImage2D;
        imageBarrier.oldLayout                   = imageInfo.initialLayout;
        imageBarrier.newLayout                   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageBarrier.srcAccessMask               = VK_ACCESS_2_NONE;
        imageBarrier.dstAccessMask               = VK_ACCESS_2_SHADER_READ_BIT;
        imageBarrier.srcStageMask                = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        imageBarrier.dstStageMask                = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBarrier.subresourceRange.baseArrayLayer = 0;
        imageBarrier.subresourceRange.baseMipLevel   = 0;
        imageBarrier.subresourceRange.layerCount     = 1;
        imageBarrier.subresourceRange.levelCount     = 1;
        imageBarrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        imageBarrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;

        VkDependencyInfo dependencyInfo        = {};
        dependencyInfo.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.imageMemoryBarrierCount = 1;
        dependencyInfo.pImageMemoryBarriers    = &imageBarrier;

        vkCmdPipelineBarrier2(cmd.transitionBuffer, &dependencyInfo);

        Submit(cmd);
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

    // Initialize command pools
    {
        int numProcessors = OS_NumProcessors();
        commandPools      = StaticArray<CommandPool>(arena, numProcessors);
    }
}

void Vulkan::AllocateCommandBuffers(ThreadCommandPool &pool, QueueType type)
{
    auto *node = pool.buffers.AddNode(ThreadCommandPool::commandBufferPoolSize);
    VkCommandBufferAllocateInfo bufferInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    bufferInfo.commandPool                 = pool.pool[i];
    bufferInfo.commandBufferCount          = ThreadCommandPool::commandBufferPoolSize;
    bufferInfo.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    VK_CHECK(vkAllocateCommandBuffers(device, &bufferInfo, node->values));
}

void Vulkan::CheckInitializedThreadCommandPool(int threadIndex)
{
    ThreadCommandPool &pool = &commandPools[threadIndex];
    if (pool.arena == 0)
    {
        pool.arena = ArenaAlloc();
        pool.pool  = StaticArray<VkCommandPool>(pool.arena, QueueType_Count, QueueType_Count);
        pool.buffers  = CommandBufferPool(pool.arena, QueueType_Count, QueueType_Count);
        pool.freeList = CommandBufferPool(pool.arena, QueueType_Count, QueueType_Count);

        for (int i = 0; i < QueueType_Count; i++)
        {
            if (families[i] != VK_QUEUE_FAMILY_IGNORED)
            {
                VkCommandPoolCreateInfo poolInfo = {
                    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
                poolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
                poolInfo.queueFamilyIndex = families[i];
                VK_CHECK(vkCreateCommandPool(device, &poolInfo, 0, &pool.pool[i]));

                AllocateCommandBuffers(pool, i);
            }
        }
    }
}

VkCommandBuffer BeginCommandBuffer(QueueType queue)
{
    int threadIndex         = GetThreadIndex();
    ThreadCommandPool &pool = commandPools[threadIndex];

    VkCommandBuffer buffer = VK_NULL_HANDLE;
    buffer                 = pool.buffers[queue].Pop();
    if (buffer == VK_NULL_HANDLE)
    {
        AllocateCommandBuffers(pool, queue);
        buffer = pool.buffers[queue].Pop();
    }

    vkResetCommandBuffer(buffer, 0);
    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(buffer, &beginInfo);
}

GPUBuffer Vulkan::TransferData(CommandBuffer buffer, size_t totalSize, int numBuffers,
                               void (*copy)(void *, u32 *))
{
    int threadIndex = GetThreadIndex();
    CheckInitializedThreadCommandPool(threadIndex);

    VkBufferCreateInfo createInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    createInfo.size               = size;
    createInfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (families.size() > 1)
    {
        createInfo.sharingMode           = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = (u32)families.size();
        createInfo.pQueueFamilyIndices   = families.data();
    }
    else
    {
        createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.usage                   = VMA_MEMORY_USAGE_AUTO;

    VkBuffer temp;
    VK_CHECK(vmaCreateBuffer(allocator, &createInfo, &allocCreateInfo, temp,
                             &buffer->allocation, 0));

    for (int i = 0; i < numBuffers; i++)
    {
        createInfo.size = createInfo.size;
        VkBuffer result;
        vkCreateBuffer(device, &createInfo, 0, &result);
                       const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer);
                       vkMapMemory(device, VkDeviceMemory memory, VkDeviceSize offset,
                                   VkDeviceSize size, VkMemoryMapFlags flags, void **ppData);
    }

    VK_CHECK(vkEndCommandBuffer(cmd.cmdBuffer));
    VK_CHECK(vkEndCommandBuffer(cmd.transitionBuffer));

    VkCommandBufferSubmitInfo bufSubmitInfo = {};
    bufSubmitInfo.sType                     = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;

    VkSemaphoreSubmitInfo waitSemInfo = {};
    waitSemInfo.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;

    VkSemaphoreSubmitInfo submitSemInfo = {};
    submitSemInfo.sType                 = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;

    VkSubmitInfo2 submitInfo = {};
    submitInfo.sType         = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;

    // Submit the copy command to the transfer queue.
    {
        bufSubmitInfo.commandBuffer = cmd.cmdBuffer;

        submitSemInfo.semaphore = cmd.semaphores[0];
        submitSemInfo.value     = 0;
        submitSemInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos    = &bufSubmitInfo;

        submitInfo.signalSemaphoreInfoCount = 1;
        submitInfo.pSignalSemaphoreInfos    = &submitSemInfo;

        MutexScope(&queues[QueueType_Copy].lock)
        {
            VK_CHECK(
                vkQueueSubmit2(queues[QueueType_Copy].queue, 1, &submitInfo, VK_NULL_HANDLE));
        }
    }
    // Insert the execution dependency (semaphores) and memory dependency (barrier) on the
    // graphics queue
    {
        bufSubmitInfo.commandBuffer = cmd.transitionBuffer;

        waitSemInfo.semaphore = cmd.semaphores[0];
        waitSemInfo.value     = 0;
        waitSemInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

        submitSemInfo.semaphore = cmd.semaphores[1];
        submitSemInfo.value     = 0;
        submitSemInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

        submitInfo.commandBufferInfoCount   = 1;
        submitInfo.pCommandBufferInfos      = &bufSubmitInfo;
        submitInfo.waitSemaphoreInfoCount   = 1;
        submitInfo.pWaitSemaphoreInfos      = &waitSemInfo;
        submitInfo.signalSemaphoreInfoCount = 1;
        submitInfo.pSignalSemaphoreInfos    = &submitSemInfo;

        MutexScope(&queues[QueueType_Graphics].lock)
        {
            VK_CHECK(vkQueueSubmit2(queues[QueueType_Graphics].queue, 1, &submitInfo,
                                    VK_NULL_HANDLE));
        }
    }
    // Execution dependency on compute queue
    {
        waitSemInfo.semaphore = cmd.semaphores[1];
        waitSemInfo.value     = 0;
        waitSemInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

        submitInfo.commandBufferInfoCount   = 0;
        submitInfo.pCommandBufferInfos      = 0;
        submitInfo.waitSemaphoreInfoCount   = 1;
        submitInfo.pWaitSemaphoreInfos      = &waitSemInfo;
        submitInfo.signalSemaphoreInfoCount = 0;
        submitInfo.pSignalSemaphoreInfos    = 0;

        MutexScope(&queues[QueueType_Compute].lock)
        {
            VK_CHECK(vkQueueSubmit2(queues[QueueType_Compute].queue, 1, &submitInfo,
                                    ToInternal(&cmd.fence)->fence));
        }
    }
    MutexScope(&mTransferMutex) { transferFreeList.push_back(cmd); }
    // TODO: compute
}

void Vulkan::CreateBufferCopy(GPUBuffer *inBuffer, GPUBufferDesc inDesc,
                              CopyFunction initCallback)
{
    GPUBufferVulkan *buffer = 0;
    MutexScope(&arenaMutex)
    {
        buffer = freeBuffer;
        if (buffer)
        {
            StackPop(freeBuffer);
        }
        else
        {
            buffer = PushStruct(arena, GPUBufferVulkan);
        }
    }

    buffer->subresourceSrv  = -1;
    buffer->subresourceUav  = -1;
    inBuffer->internalState = buffer;
    inBuffer->desc          = inDesc;
    inBuffer->mappedData    = 0;
    inBuffer->resourceType  = GPUResource::ResourceType::Buffer;
    inBuffer->ticket.ticket = 0;

    VkBufferCreateInfo createInfo = {};
    createInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    createInfo.size               = inBuffer->desc.size;

    if (HasFlags(inDesc.resourceUsage, ResourceUsage_Vertex))
    {
        createInfo.usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    }
    if (HasFlags(inDesc.resourceUsage, ResourceUsage_Index))
    {
        createInfo.usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    }

    if (HasFlags(inDesc.resourceUsage, ResourceUsage_StorageTexel))
    {
        createInfo.usage |= VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
    }
    if (HasFlags(inDesc.resourceUsage, ResourceUsage_StorageBuffer) ||
        HasFlags(inDesc.resourceUsage, ResourceUsage_StorageBufferRead))
    {
        createInfo.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    if (HasFlags(inDesc.resourceUsage, ResourceUsage_UniformTexel))
    {
        createInfo.usage |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
    }
    if (HasFlags(inDesc.resourceUsage, ResourceUsage_UniformBuffer))
    {
        createInfo.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    }

    if (HasFlags(inDesc.resourceUsage, ResourceUsage_Indirect))
    {
        createInfo.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    }
    if (HasFlags(inDesc.resourceUsage, ResourceUsage_TransferSrc))
    {
        createInfo.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    }
    if (HasFlags(inDesc.resourceUsage, ResourceUsage_TransferDst))
    {
        createInfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }

    // Sharing
    if (families.size() > 1)
    {
        createInfo.sharingMode           = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = (u32)families.size();
        createInfo.pQueueFamilyIndices   = families.data();
    }
    else
    {
        createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VmaAllocationCreateInfo allocCreateInfo = {};
    allocCreateInfo.usage                   = VMA_MEMORY_USAGE_AUTO;

    if (inDesc.usage == MemoryUsage::CPU_TO_GPU)
    {
        allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT;
        createInfo.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    }
    else if (inDesc.usage == MemoryUsage::GPU_TO_CPU)
    {
        allocCreateInfo.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        createInfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT; // TODO: not necessary?
    }

    // Buffers only on GPU must be copied to using a staging buffer
    else if (inDesc.usage == MemoryUsage::GPU_ONLY)
    {
        createInfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }

    VK_CHECK(vmaCreateBuffer(allocator, &createInfo, &allocCreateInfo, &buffer->buffer,
                             &buffer->allocation, 0));

    // Map the buffer if it's a staging buffer
    if (inDesc.usage == MemoryUsage::CPU_TO_GPU || inDesc.usage == MemoryUsage::GPU_TO_CPU)
    {
        inBuffer->mappedData = buffer->allocation->GetMappedData();
        inBuffer->desc.size  = buffer->allocation->GetSize();
    }

    if (initCallback != 0)
    {
        TransferCommand cmd;
        void *mappedData = 0;
        if (inBuffer->desc.usage == MemoryUsage::CPU_TO_GPU)
        {
            mappedData = inBuffer->mappedData;
        }
        else
        {
            cmd        = Stage(inBuffer->desc.size);
            mappedData = cmd.ringAllocation->mappedData;
        }

        initCallback(mappedData);

        if (cmd.IsValid())
        {
            if (inBuffer->desc.size != 0)
            {
                // Memory copy data to the staging buffer
                VkBufferCopy bufferCopy = {};
                bufferCopy.srcOffset    = cmd.ringAllocation->offset;
                bufferCopy.dstOffset    = 0;
                bufferCopy.size         = inBuffer->desc.size;

                RingAllocator *ringAllocator =
                    &stagingRingAllocators[cmd.ringAllocation->ringId];

                // Copy from the staging buffer to the allocated buffer
                vkCmdCopyBuffer(cmd.cmdBuffer,
                                ToInternal(&ringAllocator->transferRingBuffer)->buffer,
                                buffer->buffer, 1, &bufferCopy);
            }
            FenceVulkan *fenceVulkan = ToInternal(&cmd.fence);
            inBuffer->ticket.fence   = cmd.fence;
            inBuffer->ticket.ticket  = fenceVulkan->count;
            Submit(cmd);
        }
    }

    if (!HasFlags(inDesc.resourceUsage, ResourceUsage_Bindless))
    {
        GPUBufferVulkan::Subresource subresource;
        subresource.info.buffer = buffer->buffer;
        subresource.info.offset = 0;
        subresource.info.range  = VK_WHOLE_SIZE;
        buffer->subresources.push_back(subresource);

        // TODO: is this fine that they reference the same subresource?
        if (HasFlags(inDesc.resourceUsage, ResourceUsage_StorageTexel) ||
            HasFlags(inDesc.resourceUsage, ResourceUsage_StorageBuffer))
        {
            buffer->subresourceUav = 0;
        }
        if (HasFlags(inDesc.resourceUsage, ResourceUsage_UniformBuffer) ||
            HasFlags(inDesc.resourceUsage, ResourceUsage_UniformTexel) ||
            HasFlags(inDesc.resourceUsage, ResourceUsage_StorageBufferRead))
        {
            buffer->subresourceSrv = 0;
        }
        if (HasFlags(inDesc.resourceUsage, ResourceUsage_StorageTexel) ||
            HasFlags(inDesc.resourceUsage, ResourceUsage_UniformTexel))
        {
            Assert(0);
        }
    }
    else
    {
        Assert(!HasFlags(inDesc.resourceUsage, ResourceUsage_UniformBuffer));
        i32 subresourceIndex = -1;
        if (HasFlags(inDesc.resourceUsage, ResourceUsage_StorageTexel) ||
            HasFlags(inDesc.resourceUsage, ResourceUsage_StorageBuffer))
        {
            subresourceIndex       = CreateSubresource(inBuffer, ResourceViewType::UAV);
            buffer->subresourceUav = subresourceIndex;
        }
        if (HasFlags(inDesc.resourceUsage, ResourceUsage_UniformTexel) ||
            HasFlags(inDesc.resourceUsage, ResourceUsage_StorageBufferRead))
        {
            subresourceIndex       = CreateSubresource(inBuffer, ResourceViewType::SRV);
            buffer->subresourceSrv = subresourceIndex;
        }
    }
}

void mkGraphicsVulkan::SetName(GPUResource *resource, string name)
{
    SetName(resource, (char *)name.str);
}

void mkGraphicsVulkan::SetName(u64 handle, VkObjectType type, const char *name)
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

void Vulkan::SetName(VkQueue handle, const char *name)
{
    SetName((u64)handle, VK_OBJECT_TYPE_QUEUE, name);
}

void Vulkan::CreateRayTracingPipeline(u32 maxDepth)
{

    enum RayShaderType
    {
        RST_Raygen,
        RST_Miss,
        RST_ClosestHit,
        RST_Intersection,
        RST_Max,
    };

    std::vector<VkPipelineShaderStageCreateInfo> pipelineInfos(RST_Max);
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups(RST_Max);
    // Create pipeline infos
    {
        VkPipelineShaderStageCreateInfo info = {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.pName                = "main";
        info.stage                = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        info.module               = ;
        pipelineInfos[RST_Raygen] = info;

        info.stage              = VK_SHADER_STAGE_MISS_BIT_KHR;
        info.module             = ;
        pipelineInfos[RST_Miss] = info;

        info.stage                    = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        info.module                   = ;
        pipelineInfos[RST_ClosestHit] = info;

        info.stage                      = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
        info.module                     = ;
        pipelineInfos[RST_Intersection] = info;

        SetName(info.module, ?);
    }
    // Create shader groups
    {
        VkRayTracingShaderGroupCreateInfoKHR group = {
            VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
        group.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        group.generalShader      = VK_SHADER_UNUSED_KHR;
        group.closestHitShader   = VK_SHADER_UNUSED_KHR;
        group.anyHitShader       = VK_SHADER_UNUSED_KHR;
        group.intersectionShader = VK_SHADER_UNUSED_KHR;

        group.generalShader = RST_Raygen;
        shaderGroups.push_back(group);

        group.generalShader = RST_Miss;
        shaderGroups.push_back(group);

        group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        group.generalShader    = VK_SHADER_UNUSED_KH;
        group.closestHitShader = RST_ClosestHit;
        shaderGroups.push_back(group);

        // Intersection shader
        group.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
        group.closestHitShader   = VK_SHADER_UNUSED_KHR;
        group.intersectionShader = RST_Intersection;
        shaderGroups.push_back(group);
    }

    // Create pipeline
    {
        VkPipelineLayoutCreateInfo layoutCreateInfo = {
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};

        VkRayTracingPipelineCreateInfoKHR pipelineInfo = {
            VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};

        pipelineInfo.pStages                      = stages.data();
        pipelineInfo.stageCount                   = static_cast<u32>(stages.size());
        pipelineInfo.pGroups                      = shaderGroups.data();
        pipelineInfo.groupCount                   = static_cast<u32>(shaderGroups.size());
        pipelineInfo.maxPipelineRayRecursionDepth = maxDepth;
        pipelineInfo.layout                       = ? ;

        {
            VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV clusterInfo{
                VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV};
            clusterInfo.allowClusterAccelerationStructure = true;
            pipelineInfo.pNext                            = &clusterInfo;
        }

        VkResult result = vkCreateRayTracingPipelinesKHR(device, {}, {}, 1, &pipelineInfo, 0,
                                                         need a pipeline here);
    }
    Assert(result == VK_SUCCESS);

    // Create shader binding table
    {
    }

    // VkAccelerationStructureGeometryInstancesDataKHR instance;
    // instance.arrayOfPointers

    // Build acceleration structures, trace rays
}

GPUBVH *Vulkan::CreateBLAS(CommandList cmd, const GPUMesh *meshes, int count)
{
    ScratchArena temp;

    VkStructureType sType;
    const void *pNext;
    VkGeometryTypeKHR geometryType;
    VkAccelerationStructureGeometryDataKHR geometry;
    VkGeometryFlagsKHR flags;

    VkAccelerationStructureGeometryTrianglesDataKHR data = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    data.transformData;
    data.indexData;

    StaticArray<VkAccelerationStructureGeometryKHR> geometries(temp.temp.arena, count);
    StaticArray<VkAccelerationStructureBuildRangeInfoKHR> buildRanges(temp.temp.arena, count);
    StaticArray<u32> maxPrimitiveCounts(temp.temp.arena, count);

    VkAccelerationStructureGeometryKHR geometry = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    auto &triangles       = geometry.geometry.triangles;
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;

    for (int i = 0; i < count; i++)
    {
        GPUMesh &mesh                   = meshes[i];
        VkBufferDeviceAddressInfo pInfo = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        pInfo.buffer                    = ToInternal(buffer)->buffer;

        VkDeviceAddress vertexAddress = vkGetBufferDeviceAddress(device, &pInfo);

        pInfo.buffer                 = ToInternal(buffer)->buffer;
        VkDeviceAddress indexAddress = vkGetBufferDeviceAddress(device, &pInfo);

        triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = vertexAddress;
        triangles.vertexStride             = sizeof(Vec3f);
        triangles.maxVertex                = mesh.mesh.numVertices - 1;
        triangles.indexType                = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress  = indexAddress;

        geometries.Push(geometry);

        int primitiveCount                                 = mesh.mesh.numIndices / 3;
        VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_RANGE_INFO};
        rangeInfo.primitiveCount = primitiveCount;
        buildRanges.Push(rangeInfo);

        maxPrimitivesCounts
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.pGeometries   = geometries.data();
    buildInfo.geometryCount = static_cast<u32>(geometries.size());

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};

    vkGetAccelerationStructureBuildSizesKHR(device,
                                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &buildInfo, maxPrimitiveCounts.data, &sizeInfo);

    GPUBufferDesc desc;
    desc.size          = sizeInfo.buildScratchSize;
    desc.usage         = MemoryUsage::GPU_ONLY;
    desc.resourceUsage = ResourceUsage::;

    GPUBuffer scratch;
    CreateBuffer(&scratch, desc);

    VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer                    = ToInternal(scratch)->buffer;

    GPUBVH bvh;
    GPUBVHVulkan *bvhVulkan = PushStruct(arena, GPUBufferVulkan);

    VkDeviceAddress scratchAddress      = vkGetBufferDeviceAddress(device, &info);
    buildInfo.scratchData.deviceAddress = scratchAddress;
    buildInfo.dstAccelerationStructure  = as;

    CommandListVulkan *cmdVulkan = ToInternal(cmd);
    vkCmdBuildAccelerationStructuresKHR(cmdVulkan->GetCommandBuffer(), 1, &buildInfo,
                                        &buildRanges.data());

    bvh.internalState = bvhVulkan;
    return bvh;
}

} // namespace rt
