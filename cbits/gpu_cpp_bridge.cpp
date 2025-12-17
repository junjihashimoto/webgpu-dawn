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
#include <cstring>
#include <stdexcept>
#include <memory>
#include <iostream>

// Enable logging
#define GPU_DEBUG_LOG 1

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
};

// Convert between NumType enums
static gpu::NumType to_cpp_numtype(GPUNumType dtype) {
    switch (dtype) {
        case GPU_F16: return gpu::kf16;
        case GPU_F32: return gpu::kf32;
        case GPU_F64: return gpu::kf64;
        case GPU_I8:  return gpu::ki8;
        case GPU_I16: return gpu::ki16;
        case GPU_I32: return gpu::ki32;
        case GPU_I64: return gpu::ki64;
        case GPU_U8:  return gpu::ku8;
        case GPU_U16: return gpu::ku16;
        case GPU_U32: return gpu::ku32;
        case GPU_U64: return gpu::ku64;
        default:      return gpu::kUnknown;
    }
}

extern "C" {

// Context management
GPUContext gpu_create_context(GPUError* error) {
    try {
        CLEAR_ERROR(error);
        auto ctx = new GPUContextImpl();
        ctx->ctx = new gpu::Context(gpu::createContext());
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

        // Create tensor based on dtype
        gpu::NumType cpp_dtype = to_cpp_numtype(dtype);
        switch (dtype) {
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
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
    }
}

void gpu_to_gpu(GPUContext ctx, const void* input, GPUTensor tensor, size_t size, GPUError* error) {
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
                            GPUError* error) {
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
        LOG("Calling gpu::createKernel...");
        gpu::CompilationInfo compilationInfo = {};
        auto impl = new GPUKernelImpl();
        impl->kernel = new gpu::Kernel(
            gpu::createKernel(*ctx_impl->ctx, *code_impl->code,
                            cpp_tensors.data(), num_tensors,
                            viewOffsets.data(),
                            workgroups,
                            nullptr, 0,  // params, paramsSize
                            &compilationInfo)  // compilation info
        );

        // Check for compilation errors
        if (!compilationInfo.messages.empty()) {
            LOG("Shader compilation messages:");
            for (size_t i = 0; i < compilationInfo.messages.size(); ++i) {
                LOG("  Line " << compilationInfo.lineNums[i] << ":" << compilationInfo.linePos[i] << " - " << compilationInfo.messages[i]);
            }
        }

        LOG("Kernel created successfully");
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

} // extern "C"
