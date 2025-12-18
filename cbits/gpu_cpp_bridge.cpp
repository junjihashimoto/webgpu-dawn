/*
 * C++ bridge for Haskell FFI to gpu.cpp
 *
 * This file uses gpu.hpp from the gpu.cpp project:
 * https://github.com/AnswerDotAI/gpu.cpp
 *
 * gpu.cpp is licensed under Apache License 2.0
 * See THIRD_PARTY_LICENSES.md for full license text
 */

#include "gpu_wrapper.h"
#include "gpu.hpp"
#include "webgpu.h"
#include <cstring>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <chrono>
#include <atomic>

// WebGPU buffer limits for large models - matches gpu.cpp's LIMITS_BUFFER_SIZE_1GB
// See https://github.com/google/dawn/blob/main/src/dawn/native/Limits.cpp
#define GPU_LIMITS_1GB { \
    .nextInChain = nullptr, \
    .maxTextureDimension1D = 8192, \
    .maxTextureDimension2D = 8192, \
    .maxTextureDimension3D = 2048, \
    .maxTextureArrayLayers = 256, \
    .maxBindGroups = 4, \
    .maxBindGroupsPlusVertexBuffers = 24, \
    .maxBindingsPerBindGroup = 1000, \
    .maxDynamicUniformBuffersPerPipelineLayout = 8, \
    .maxDynamicStorageBuffersPerPipelineLayout = 4, \
    .maxSampledTexturesPerShaderStage = 16, \
    .maxSamplersPerShaderStage = 16, \
    .maxStorageBuffersPerShaderStage = 10, \
    .maxStorageTexturesPerShaderStage = 4, \
    .maxUniformBuffersPerShaderStage = 12, \
    .maxUniformBufferBindingSize = 65536, \
    .maxStorageBufferBindingSize = 0x600000000, \
    .minUniformBufferOffsetAlignment = 256, \
    .minStorageBufferOffsetAlignment = 256, \
    .maxVertexBuffers = 8, \
    .maxBufferSize = 0x600000000, \
    .maxVertexAttributes = 16, \
    .maxVertexBufferArrayStride = 2048, \
    .maxInterStageShaderVariables = 16, \
    .maxColorAttachments = 8, \
    .maxColorAttachmentBytesPerSample = 32, \
    .maxComputeWorkgroupStorageSize = 16384, \
    .maxComputeInvocationsPerWorkgroup = 256, \
    .maxComputeWorkgroupSizeX = 256, \
    .maxComputeWorkgroupSizeY = 256, \
    .maxComputeWorkgroupSizeZ = 64, \
    .maxComputeWorkgroupsPerDimension = 65535, \
    .maxImmediateSize = 0 \
  }

// ============================================================================
// PROFILING INSTRUMENTATION
// ============================================================================

#define ENABLE_PROFILING 1

#if ENABLE_PROFILING
#include <map>
#include <mutex>

struct ProfileStats {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> total_us{0};  // microseconds
};

static std::map<std::string, ProfileStats> g_profile_stats;
static std::mutex g_profile_mutex;

#define PROFILE_START() auto __prof_start = std::chrono::high_resolution_clock::now()
#define PROFILE_END(category) do { \
    auto __prof_end = std::chrono::high_resolution_clock::now(); \
    auto __prof_us = std::chrono::duration_cast<std::chrono::microseconds>(__prof_end - __prof_start).count(); \
    g_profile_stats[category].count.fetch_add(1); \
    g_profile_stats[category].total_us.fetch_add(__prof_us); \
} while(0)

// Export profiling stats
extern "C" void gpu_print_profile_stats() {
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    std::cerr << "\n=== GPU PROFILING STATS ===" << std::endl;
    uint64_t total_ops = 0;
    uint64_t total_time_us = 0;

    for (const auto& [name, stats] : g_profile_stats) {
        uint64_t count = stats.count.load();
        uint64_t time_us = stats.total_us.load();
        total_ops += count;
        total_time_us += time_us;

        if (count > 0) {
            double avg_us = (double)time_us / count;
            double total_ms = time_us / 1000.0;
            std::cerr << name << ":" << std::endl;
            std::cerr << "  Count: " << count << std::endl;
            std::cerr << "  Total: " << total_ms << " ms" << std::endl;
            std::cerr << "  Avg: " << avg_us << " us" << std::endl;
        }
    }

    std::cerr << "\nTOTAL OPERATIONS: " << total_ops << std::endl;
    std::cerr << "TOTAL TIME: " << (total_time_us / 1000.0) << " ms" << std::endl;
    std::cerr << "========================\n" << std::endl;
}

extern "C" void gpu_reset_profile_stats() {
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    g_profile_stats.clear();
}

#else
#define PROFILE_START()
#define PROFILE_END(category)
#endif

// Enable logging (disabled for performance)
#define GPU_DEBUG_LOG 1  // ENABLED: Debug GPU errors

#if GPU_DEBUG_LOG
#define LOG(msg) do { std::cerr << "[GPU] " << msg << std::endl; std::cerr.flush(); } while(0)
#else
#define LOG(msg)
#endif

// Helper macro for error handling
#define SET_ERROR(err, code_val, msg) \
    if (err) { \
        err->code = code_val; \
        err->message = msg; \
        LOG("ERROR: " << msg); \
    }

#define CLEAR_ERROR(err) \
    if (err) { \
        err->code = 0; \
        err->message = nullptr; \
    }

// Wrappers for C++ objects
struct GPUContextImpl {
    gpu::Context* ctx;
    std::vector<std::future<void>> pendingDispatches;  // For async execution
};

struct GPUTensorImpl {
    gpu::Tensor* tensor;
};

struct GPUShapeImpl {
    gpu::Shape* shape;
};

struct GPUKernelCodeImpl {
    gpu::KernelCode* code;
};

struct GPUKernelImpl {
    gpu::Kernel* kernel;
    std::string entryPoint;  // For profiling categorization
};

// Convert between NumType enums
static gpu::NumType to_cpp_numtype(GPUNumType dtype) {
    switch (dtype) {
        case GPU_F16: return gpu::kf16;
        case GPU_F32: return gpu::kf32;
        case GPU_F64: return gpu::kf64;
        case GPU_I4:  return gpu::ki4;
        case GPU_I8:  return gpu::ki8;
        case GPU_I16: return gpu::ki16;
        case GPU_I32: return gpu::ki32;
        case GPU_I64: return gpu::ki64;
        case GPU_U4:  return gpu::ku4;
        case GPU_U8:  return gpu::ku8;
        case GPU_U16: return gpu::ku16;
        case GPU_U32: return gpu::ku32;
        case GPU_U64: return gpu::ku64;
        default:      return gpu::kUnknown;
    }
}

extern "C" {

// Context management
// WebGPU error callback to catch out-of-memory and other fatal errors
static void webgpuErrorCallback(WGPUErrorType type, char const * message, void * userdata) {
    const char* errorType = "Unknown";
    switch (type) {
        case WGPUErrorType_NoError: errorType = "NoError"; break;
        case WGPUErrorType_Validation: errorType = "Validation"; break;
        case WGPUErrorType_OutOfMemory: errorType = "OutOfMemory"; break;
        case WGPUErrorType_Internal: errorType = "Internal"; break;
        case WGPUErrorType_Unknown: errorType = "Unknown"; break;
        default: break;
    }

    std::cerr << "❌ FATAL WebGPU ERROR [" << errorType << "]: " << message << std::endl;

    // For out-of-memory errors, provide helpful guidance
    if (type == WGPUErrorType_OutOfMemory) {
        std::cerr << "\n💡 OUT OF MEMORY GUIDANCE:" << std::endl;
        std::cerr << "   - Model size (~4.9 GB) may exceed available GPU memory" << std::endl;
        std::cerr << "   - Try reducing batch size or sequence length" << std::endl;
        std::cerr << "   - Consider using a smaller model" << std::endl;
        std::cerr << "   - Check system GPU memory availability\n" << std::endl;
        exit(1);  // Exit immediately on OOM
    }
}

GPUContext gpu_create_context(GPUError* error) {
    PROFILE_START();
    try {
        CLEAR_ERROR(error);
        auto ctx = new GPUContextImpl();

        // Use default createContext() with no arguments
        // This will trigger automatic feature and limits detection in gpu.hpp
        ctx->ctx = new gpu::Context(gpu::createContext());

        // Query actual device limits to verify what was granted
        WGPULimits actualLimits = {};
        WGPUStatus status = wgpuDeviceGetLimits(ctx->ctx->device, &actualLimits);

        std::cerr << "✅ GPU context created" << std::endl;
        if (status == WGPUStatus_Success) {
            std::cerr << "   ACTUAL maxStorageBufferBindingSize: "
                      << (actualLimits.maxStorageBufferBindingSize / (1024*1024)) << " MB (0x"
                      << std::hex << actualLimits.maxStorageBufferBindingSize << std::dec << ")" << std::endl;
            std::cerr << "   ACTUAL maxBufferSize: "
                      << (actualLimits.maxBufferSize / (1024*1024)) << " MB (0x"
                      << std::hex << actualLimits.maxBufferSize << std::dec << ")" << std::endl;
        } else {
            std::cerr << "   ERROR: Failed to query device limits!" << std::endl;
        }

        PROFILE_END("create_context");
        return static_cast<GPUContext>(ctx);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

// Create context with device features and toggles
GPUContext gpu_create_context_with_features(
    const char** enabledToggles, size_t toggleCount,
    const uint32_t* requiredFeatures, size_t featureCount,
    GPUError* error) {
    try {
        CLEAR_ERROR(error);
        LOG("Creating context with " << toggleCount << " toggles and " << featureCount << " features");

        // Setup Dawn toggles
        WGPUDawnTogglesDescriptor toggles = {};
        toggles.chain.sType = WGPUSType_DawnTogglesDescriptor;
        toggles.enabledToggles = enabledToggles;
        toggles.enabledToggleCount = toggleCount;

        // Setup device descriptor with features
        WGPUDeviceDescriptor devDesc = {};
        devDesc.nextInChain = &toggles.chain;
        devDesc.requiredFeatureCount = featureCount;

        // Convert uint32_t features to WGPUFeatureName
        std::vector<WGPUFeatureName> features(featureCount);
        for (size_t i = 0; i < featureCount; ++i) {
            features[i] = static_cast<WGPUFeatureName>(requiredFeatures[i]);
            LOG("  Requesting feature: " << features[i]);
        }
        devDesc.requiredFeatures = features.data();

        // Error callback - ALWAYS output errors to stderr
        devDesc.uncapturedErrorCallbackInfo = WGPUUncapturedErrorCallbackInfo {
            .callback = [](WGPUDevice const * device, WGPUErrorType type, WGPUStringView msg, void*, void*) {
                // CRITICAL: Always output GPU errors regardless of LOG macro
                std::cerr << "[GPU ERROR] Type=" << (int)type << " Message: " << std::string(msg.data, msg.length) << std::endl;
                std::cerr.flush();
            }
        };

        // Device lost callback - ALWAYS output to stderr
        devDesc.deviceLostCallbackInfo = WGPUDeviceLostCallbackInfo {
            .mode = WGPUCallbackMode_AllowSpontaneous,
            .callback = [](WGPUDevice const * device, WGPUDeviceLostReason reason, WGPUStringView msg, void*, void*) {
                // CRITICAL: Always output device lost errors
                std::cerr << "[GPU DEVICE LOST] Reason=" << (int)reason << " Message: " << std::string(msg.data, msg.length) << std::endl;
                std::cerr.flush();
            }
        };

        // Create context with custom device descriptor
        LOG("Calling gpu::createContext with device descriptor...");
        auto ctx = new GPUContextImpl();
        ctx->ctx = new gpu::Context(gpu::createContext({}, {}, devDesc));

        // Setup logging callback
        WGPULoggingCallbackInfo logCb{
            .callback = [](WGPULoggingType type, WGPUStringView msg, void*, void*) {
                LOG("[WGPU " << (int)type << "] " << std::string(msg.data, msg.length));
            }
        };
        wgpuDeviceSetLoggingCallback(ctx->ctx->device, logCb);

        // Log available features on the device
        LOG("Device created. Checking supported features...");
        WGPUSupportedFeatures supportedFeatures = {};
        wgpuDeviceGetFeatures(ctx->ctx->device, &supportedFeatures);
        LOG("Device supports " << supportedFeatures.featureCount << " features");
        if (supportedFeatures.featureCount > 0 && supportedFeatures.features != nullptr) {
            LOG("Supported features:");
            for (size_t i = 0; i < supportedFeatures.featureCount; ++i) {
                LOG("  Feature " << i << ": " << supportedFeatures.features[i]);
            }
        }

        LOG("Context created successfully with features");
        return static_cast<GPUContext>(ctx);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

void gpu_destroy_context(GPUContext ctx) {
    if (ctx) {
        auto impl = static_cast<GPUContextImpl*>(ctx);
        delete impl->ctx;
        delete impl;
    }
}

// Shape management
GPUShape gpu_create_shape(const size_t* dims, size_t rank, GPUError* error) {
    try {
        CLEAR_ERROR(error);
        if (rank > 8) {
            SET_ERROR(error, 2, "Shape rank cannot exceed 8");
            return nullptr;
        }
        auto impl = new GPUShapeImpl();
        impl->shape = new gpu::Shape();
        impl->shape->rank = rank;
        for (size_t i = 0; i < rank; ++i) {
            impl->shape->data[i] = dims[i];
        }
        return static_cast<GPUShape>(impl);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

void gpu_destroy_shape(GPUShape shape) {
    if (shape) {
        auto impl = static_cast<GPUShapeImpl*>(shape);
        delete impl->shape;
        delete impl;
    }
}

size_t gpu_shape_size(GPUShape shape) {
    if (!shape) return 0;
    auto impl = static_cast<GPUShapeImpl*>(shape);
    return gpu::size(*impl->shape);
}

size_t gpu_shape_rank(GPUShape shape) {
    if (!shape) return 0;
    auto impl = static_cast<GPUShapeImpl*>(shape);
    return impl->shape->rank;
}

size_t gpu_shape_dim(GPUShape shape, size_t index) {
    if (!shape) return 0;
    auto impl = static_cast<GPUShapeImpl*>(shape);
    if (index >= impl->shape->rank) return 0;
    return (*impl->shape)[index];
}

// Tensor management
GPUTensor gpu_create_tensor(GPUContext ctx, GPUShape shape, GPUNumType dtype, GPUError* error) {
    try {
        CLEAR_ERROR(error);
        if (!ctx || !shape) {
            SET_ERROR(error, 3, "Invalid context or shape");
            return nullptr;
        }
        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);
        auto shape_impl = static_cast<GPUShapeImpl*>(shape);

        auto impl = new GPUTensorImpl();
        impl->tensor = new gpu::Tensor(
            gpu::createTensor(*ctx_impl->ctx, *shape_impl->shape, to_cpp_numtype(dtype))
        );
        return static_cast<GPUTensor>(impl);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

// Track GPU memory usage
static std::atomic<size_t> g_total_gpu_memory{0};
static std::atomic<size_t> g_tensor_count{0};

GPUTensor gpu_create_tensor_with_data(GPUContext ctx, GPUShape shape, GPUNumType dtype,
                                       const void* data, size_t data_size, GPUError* error) {
    try {
        CLEAR_ERROR(error);
        if (!ctx || !shape || !data) {
            SET_ERROR(error, 3, "Invalid context, shape, or data");
            return nullptr;
        }
        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);
        auto shape_impl = static_cast<GPUShapeImpl*>(shape);

        auto impl = new GPUTensorImpl();

        // Track memory allocation
        g_tensor_count.fetch_add(1);
        g_total_gpu_memory.fetch_add(data_size);

        // Log large tensor allocations (> 100 MB)
        if (data_size > 100 * 1024 * 1024) {
            size_t mb = data_size / (1024 * 1024);
            size_t total_mb = g_total_gpu_memory.load() / (1024 * 1024);
            std::cerr << "🔸 Allocating large tensor: " << mb << " MB (total: " << total_mb << " MB, count: " << g_tensor_count.load() << ")" << std::endl;
        }

        // Create tensor based on dtype
        gpu::NumType cpp_dtype = to_cpp_numtype(dtype);
        switch (dtype) {
            case GPU_F16:
                impl->tensor = new gpu::Tensor(
                    gpu::createTensor(*ctx_impl->ctx, *shape_impl->shape, cpp_dtype,
                                     static_cast<const half*>(data))
                );
                break;
            case GPU_F32:
                impl->tensor = new gpu::Tensor(
                    gpu::createTensor(*ctx_impl->ctx, *shape_impl->shape, cpp_dtype,
                                     static_cast<const float*>(data))
                );
                break;
            case GPU_I32:
                impl->tensor = new gpu::Tensor(
                    gpu::createTensor(*ctx_impl->ctx, *shape_impl->shape, cpp_dtype,
                                     static_cast<const int32_t*>(data))
                );
                break;
            case GPU_U32:
                impl->tensor = new gpu::Tensor(
                    gpu::createTensor(*ctx_impl->ctx, *shape_impl->shape, cpp_dtype,
                                     static_cast<const uint32_t*>(data))
                );
                break;
            case GPU_U4:
                // U4 data is pre-packed into uint32_t arrays (8 nibbles per u32)
                impl->tensor = new gpu::Tensor(
                    gpu::createTensor(*ctx_impl->ctx, *shape_impl->shape, cpp_dtype,
                                     static_cast<const uint32_t*>(data))
                );
                break;
            case GPU_I4:
                // I4 data is pre-packed into uint32_t arrays (8 nibbles per u32)
                // Note: Using uint32_t* because the packing is the same; shader handles sign
                impl->tensor = new gpu::Tensor(
                    gpu::createTensor(*ctx_impl->ctx, *shape_impl->shape, cpp_dtype,
                                     static_cast<const uint32_t*>(data))
                );
                break;
            // Add more cases as needed
            default:
                SET_ERROR(error, 4, "Unsupported data type for tensor creation with data");
                delete impl;
                return nullptr;
        }

        return static_cast<GPUTensor>(impl);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

void gpu_destroy_tensor(GPUTensor tensor) {
    if (tensor) {
        auto impl = static_cast<GPUTensorImpl*>(tensor);
        delete impl->tensor;
        delete impl;
    }
}

size_t gpu_tensor_size_bytes(GPUTensor tensor) {
    if (!tensor) return 0;
    auto impl = static_cast<GPUTensorImpl*>(tensor);
    return impl->tensor->data.size;
}

// Data transfer
void gpu_to_cpu(GPUContext ctx, GPUTensor tensor, void* output, size_t size, GPUError* error) {
    PROFILE_START();
    try {
        LOG("Reading " << size << " bytes from GPU");
        CLEAR_ERROR(error);
        if (!ctx || !tensor || !output) {
            SET_ERROR(error, 3, "Invalid context, tensor, or output buffer");
            return;
        }
        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);
        auto tensor_impl = static_cast<GPUTensorImpl*>(tensor);

        gpu::toCPU(*ctx_impl->ctx, *tensor_impl->tensor, output, size);

        // Log first few bytes for debugging
        float* data = static_cast<float*>(output);
        LOG("First 4 values after GPU read: " << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3]);

        PROFILE_END("gpu_to_cpu");
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
    }
}

void gpu_to_gpu(GPUContext ctx, const void* input, GPUTensor tensor, size_t size, GPUError* error) {
    PROFILE_START();
    try {
        CLEAR_ERROR(error);
        if (!ctx || !tensor || !input) {
            SET_ERROR(error, 3, "Invalid context, tensor, or input buffer");
            return;
        }
        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);
        auto tensor_impl = static_cast<GPUTensorImpl*>(tensor);

        // Use toGPU with Tensor reference (gpu.cpp API)
        // Cast to appropriate type based on tensor's dtype
        gpu::toGPU(*ctx_impl->ctx, static_cast<const float*>(input),
                  *tensor_impl->tensor);
        PROFILE_END("cpu_to_gpu");
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
    }
}

// Kernel code management
GPUKernelCode gpu_create_kernel_code(const char* wgsl_code, GPUError* error) {
    try {
        CLEAR_ERROR(error);
        if (!wgsl_code) {
            SET_ERROR(error, 3, "Invalid WGSL code");
            return nullptr;
        }
        auto impl = new GPUKernelCodeImpl();
        impl->code = new gpu::KernelCode();
        impl->code->data = std::string(wgsl_code);
        return static_cast<GPUKernelCode>(impl);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

void gpu_set_kernel_workgroup_size(GPUKernelCode code, size_t x, size_t y, size_t z) {
    if (!code) return;
    auto impl = static_cast<GPUKernelCodeImpl*>(code);
    impl->code->workgroupSize = {x, y, z};
}

void gpu_set_kernel_entry_point(GPUKernelCode code, const char* entry_point) {
    if (!code || !entry_point) return;
    auto impl = static_cast<GPUKernelCodeImpl*>(code);
    impl->code->entryPoint = std::string(entry_point);
}

void gpu_destroy_kernel_code(GPUKernelCode code) {
    if (code) {
        auto impl = static_cast<GPUKernelCodeImpl*>(code);
        delete impl->code;
        delete impl;
    }
}

// Kernel compilation and execution
GPUKernel gpu_create_kernel(GPUContext ctx, GPUKernelCode code,
                            GPUTensor* tensors, size_t num_tensors,
                            size_t num_workgroups_x, size_t num_workgroups_y, size_t num_workgroups_z,
                            const char* cache_key,
                            GPUError* error) {
    PROFILE_START();
    try {
        LOG("Creating kernel with " << num_tensors << " tensors and workgroups (" << num_workgroups_x << "," << num_workgroups_y << "," << num_workgroups_z << ")");
        CLEAR_ERROR(error);
        if (!ctx || !code) {
            SET_ERROR(error, 3, "Invalid context or kernel code");
            return nullptr;
        }

        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);
        auto code_impl = static_cast<GPUKernelCodeImpl*>(code);

        // Convert tensor array to C++ vector
        std::vector<gpu::Tensor> cpp_tensors;
        std::vector<size_t> viewOffsets(num_tensors, 0);  // All offsets are 0
        for (size_t i = 0; i < num_tensors; ++i) {
            auto tensor_impl = static_cast<GPUTensorImpl*>(tensors[i]);
            cpp_tensors.push_back(*tensor_impl->tensor);
            LOG("  Tensor " << i << ": size=" << tensor_impl->tensor->data.size << " bytes, shape rank=" << tensor_impl->tensor->shape.rank);
        }

        gpu::Shape workgroups = {num_workgroups_x, num_workgroups_y, num_workgroups_z};

        LOG("Shader code entry point: " << code_impl->code->entryPoint);
        LOG("Shader code length: " << code_impl->code->data.length() << " bytes");
        LOG("Shader code:\n" << code_impl->code->data);

        // Generate cache key from shader code if user didn't provide one
        // This enables automatic shader caching for 10-20x speedup!
        // IMPORTANT: auto_cache_key must be declared OUTSIDE the if block
        // to prevent the string from being destroyed before use!
        std::string auto_cache_key;
        const char* final_cache_key = cache_key;
        if (cache_key == nullptr) {
            // Simple hash: combine shader code hash + workgroup size + TENSOR BUFFERS
            // The tensor buffers MUST be included because bind groups are bound to specific buffers!
            std::hash<std::string> hasher;
            size_t hash = hasher(code_impl->code->data);
            hash ^= hasher(code_impl->code->entryPoint) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= num_workgroups_x + num_workgroups_y * 1000 + num_workgroups_z * 1000000;

            // Include tensor buffer pointers in the hash
            // This ensures different tensors get different cached kernels (with different bind groups)
            for (size_t i = 0; i < num_tensors; ++i) {
                uintptr_t buf_ptr = reinterpret_cast<uintptr_t>(cpp_tensors[i].data.buffer);
                hash ^= buf_ptr + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }

            auto_cache_key = "shader_" + std::to_string(hash);
            final_cache_key = auto_cache_key.c_str();
            LOG("Auto-generated cache key: " << final_cache_key);
        }

        LOG("Calling gpu::createKernel with caching...");
        LOG("Cache key being passed: " << (final_cache_key ? final_cache_key : "nullptr"));
        gpu::CompilationInfo compilationInfo = {};
        auto impl = new GPUKernelImpl();
        impl->entryPoint = code_impl->code->entryPoint;  // Store for profiling
        impl->kernel = new gpu::Kernel(
            gpu::createKernel(*ctx_impl->ctx, *code_impl->code,
                            cpp_tensors.data(), num_tensors,
                            viewOffsets.data(),
                            workgroups,
                            nullptr, 0,  // params, paramsSize
                            &compilationInfo,  // compilation info
                            final_cache_key)   // ENABLE CACHING!
        );

        // Check for compilation errors
        if (!compilationInfo.messages.empty()) {
            LOG("Shader compilation messages:");
            for (size_t i = 0; i < compilationInfo.messages.size(); ++i) {
                LOG("  Line " << compilationInfo.lineNums[i] << ":" << compilationInfo.linePos[i] << " - " << compilationInfo.messages[i]);
            }
        }

        LOG("Kernel created successfully");

        // Categorize kernel by entry point for detailed profiling
        std::string entryPoint = code_impl->code->entryPoint;
        std::string category = "kernel_" + entryPoint;
        PROFILE_END(category.c_str());

        return static_cast<GPUKernel>(impl);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

void gpu_destroy_kernel(GPUKernel kernel) {
    if (kernel) {
        auto impl = static_cast<GPUKernelImpl*>(kernel);
        delete impl->kernel;
        delete impl;
    }
}

void gpu_dispatch_kernel(GPUContext ctx, GPUKernel kernel, GPUError* error) {
    PROFILE_START();
    try {
        LOG("Dispatching kernel...");
        CLEAR_ERROR(error);
        if (!ctx || !kernel) {
            SET_ERROR(error, 3, "Invalid context or kernel");
            return;
        }

        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);
        auto kernel_impl = static_cast<GPUKernelImpl*>(kernel);

        // Dispatch the kernel synchronously
        gpu::dispatchKernel(*ctx_impl->ctx, *kernel_impl->kernel);
        LOG("Kernel dispatched successfully");

        // Categorize dispatch by operation type for detailed profiling
        std::string category = "dispatch_" + kernel_impl->entryPoint;
        PROFILE_END(category.c_str());
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
    }
}

// Utility functions
size_t gpu_size_of_type(GPUNumType dtype) {
    return gpu::sizeBytes(to_cpp_numtype(dtype), 1);
}

const char* gpu_type_name(GPUNumType dtype) {
    static const char* names[] = {
        "f16", "f32", "f64",
        "i8", "i16", "i32", "i64",
        "u8", "u16", "u32", "u64",
        "unknown"
    };
    if (dtype >= 0 && dtype <= GPU_UNKNOWN) {
        return names[dtype];
    }
    return "invalid";
}

// Async dispatch - returns immediately without waiting
void gpu_dispatch_kernel_async(GPUContext ctx, GPUKernel kernel, GPUError* error) {
    PROFILE_START();
    try {
        LOG("Dispatching kernel async...");
        CLEAR_ERROR(error);
        if (!ctx || !kernel) {
            SET_ERROR(error, 3, "Invalid context or kernel");
            return;
        }

        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);
        auto kernel_impl = static_cast<GPUKernelImpl*>(kernel);

        // Dispatch asynchronously - returns future immediately
        std::future<void> future = gpu::dispatchKernelAsync(*ctx_impl->ctx, *kernel_impl->kernel);
        
        // Store future for later synchronization
        ctx_impl->pendingDispatches.push_back(std::move(future));
        
        LOG("Kernel dispatched async (not waiting)");

        // Categorize dispatch by operation type for detailed profiling
        std::string category = "dispatch_async_" + kernel_impl->entryPoint;
        PROFILE_END(category.c_str());
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
    }
}

// Wait for all pending async dispatches
void gpu_wait_all(GPUContext ctx) {
    PROFILE_START();
    try {
        if (!ctx) return;
        
        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);
        
        LOG("Waiting for " << ctx_impl->pendingDispatches.size() << " pending dispatches...");
        
        // Wait for all pending futures
        for (auto& future : ctx_impl->pendingDispatches) {
            gpu::wait(*ctx_impl->ctx, future);
        }
        
        // Clear completed futures
        ctx_impl->pendingDispatches.clear();
        
        LOG("All dispatches completed");
        PROFILE_END("wait_all");
    } catch (const std::exception& e) {
        // Can't set error here since wait_all doesn't have error parameter
        std::cerr << "Error in gpu_wait_all: " << e.what() << std::endl;
    }
}

// Begin batching GPU commands - all subsequent dispatches will be accumulated
void gpu_begin_batch(GPUContext ctx, GPUError* error) {
    PROFILE_START();
    try {
        CLEAR_ERROR(error);
        if (!ctx) {
            SET_ERROR(error, 3, "Invalid context");
            return;
        }

        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);
        gpu::beginBatch(*ctx_impl->ctx);
        PROFILE_END("begin_batch");
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
    }
}

// End batching and submit all accumulated commands in a single submission
void gpu_end_batch(GPUContext ctx, GPUError* error) {
    PROFILE_START();
    try {
        CLEAR_ERROR(error);
        if (!ctx) {
            SET_ERROR(error, 3, "Invalid context");
            return;
        }

        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);
        std::future<void> future = gpu::endBatch(*ctx_impl->ctx);

        // Store future for synchronization
        ctx_impl->pendingDispatches.push_back(std::move(future));

        PROFILE_END("end_batch");
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
    }
}

} // extern "C"
