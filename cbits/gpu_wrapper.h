#ifndef GPU_WRAPPER_H
#define GPU_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types for Haskell FFI
typedef void* GPUContext;
typedef void* GPUTensor;
typedef void* GPUKernel;
typedef void* GPUShape;
typedef void* GPUKernelCode;
typedef void* GPUQuerySet;
typedef void* GPUCommandEncoder;
typedef void* GPUBuffer;

// Numeric types enum matching gpu.hpp
typedef enum {
    GPU_F16 = 0,
    GPU_F32 = 1,
    GPU_F64 = 2,
    GPU_I4  = 3,   // 4-bit signed (packed: 8 nibbles per u32)
    GPU_I8  = 4,
    GPU_I16 = 5,
    GPU_I32 = 6,
    GPU_I64 = 7,
    GPU_U4  = 8,   // 4-bit unsigned (packed: 8 nibbles per u32)
    GPU_U8  = 9,
    GPU_U16 = 10,
    GPU_U32 = 11,
    GPU_U64 = 12,
    GPU_UNKNOWN = 13
} GPUNumType;

// Error handling
typedef struct {
    int code;
    const char* message;
} GPUError;

// Context management
GPUContext gpu_create_context(GPUError* error);
GPUContext gpu_create_context_with_features(
    const char** enabledToggles, size_t toggleCount,
    const uint32_t* requiredFeatures, size_t featureCount,
    GPUError* error);
void gpu_destroy_context(GPUContext ctx);

// Shape management
GPUShape gpu_create_shape(const size_t* dims, size_t rank, GPUError* error);
void gpu_destroy_shape(GPUShape shape);
size_t gpu_shape_size(GPUShape shape);
size_t gpu_shape_rank(GPUShape shape);
size_t gpu_shape_dim(GPUShape shape, size_t index);

// Tensor management
GPUTensor gpu_create_tensor(GPUContext ctx, GPUShape shape, GPUNumType dtype, GPUError* error);
GPUTensor gpu_create_tensor_with_data(GPUContext ctx, GPUShape shape, GPUNumType dtype,
                                       const void* data, size_t data_size, GPUError* error);
void gpu_destroy_tensor(GPUTensor tensor);
size_t gpu_tensor_size_bytes(GPUTensor tensor);

// Data transfer
void gpu_to_cpu(GPUContext ctx, GPUTensor tensor, void* output, size_t size, GPUError* error);
void gpu_to_gpu(GPUContext ctx, const void* input, GPUTensor tensor, size_t size, GPUError* error);

// Kernel code management
GPUKernelCode gpu_create_kernel_code(const char* wgsl_code, GPUError* error);
void gpu_set_kernel_workgroup_size(GPUKernelCode code, size_t x, size_t y, size_t z);
void gpu_set_kernel_entry_point(GPUKernelCode code, const char* entry_point);
void gpu_destroy_kernel_code(GPUKernelCode code);

// Kernel compilation and execution
GPUKernel gpu_create_kernel(GPUContext ctx, GPUKernelCode code,
                            GPUTensor* tensors, size_t num_tensors,
                            size_t num_workgroups_x, size_t num_workgroups_y, size_t num_workgroups_z,
                            const char* cache_key,  // Optional cache key for shader caching
                            GPUError* error);
void gpu_destroy_kernel(GPUKernel kernel);
void gpu_dispatch_kernel(GPUContext ctx, GPUKernel kernel, GPUError* error);

// Async dispatch - returns immediately, kernel executes in background
// Call gpu_wait_all() to synchronize
void gpu_dispatch_kernel_async(GPUContext ctx, GPUKernel kernel, GPUError* error);

// Wait for all pending async dispatches to complete
void gpu_wait_all(GPUContext ctx);

// Command batching - accumulate multiple kernels into a single submission
void gpu_begin_batch(GPUContext ctx, GPUError* error);
void gpu_end_batch(GPUContext ctx, GPUError* error);

// Utility functions
size_t gpu_size_of_type(GPUNumType dtype);
const char* gpu_type_name(GPUNumType dtype);

// QuerySet types for timestamp queries
typedef enum {
    GPU_QUERY_TYPE_OCCLUSION = 1,
    GPU_QUERY_TYPE_TIMESTAMP = 2
} GPUQueryType;

// Timestamp query support
GPUQuerySet gpu_create_query_set(GPUContext ctx, GPUQueryType type, uint32_t count, GPUError* error);
void gpu_destroy_query_set(GPUQuerySet querySet);

// Get raw WebGPU handles for advanced usage
GPUCommandEncoder gpu_get_command_encoder(GPUContext ctx, GPUError* error);
void gpu_release_command_encoder(GPUCommandEncoder encoder);

// Timestamp query commands (to be called on command encoder)
void gpu_write_timestamp(GPUCommandEncoder encoder, GPUQuerySet querySet, uint32_t queryIndex);
void gpu_resolve_query_set(GPUCommandEncoder encoder, GPUQuerySet querySet,
                           uint32_t firstQuery, uint32_t queryCount,
                           GPUBuffer destination, uint64_t destinationOffset);

// Create buffer for query results
GPUBuffer gpu_create_query_buffer(GPUContext ctx, size_t size, GPUError* error);
void gpu_release_buffer(GPUBuffer buffer);

// Read back query results (blocking)
void gpu_read_query_buffer(GPUContext ctx, GPUBuffer buffer, uint64_t* data, size_t count, GPUError* error);

// Profiling functions
void gpu_print_profile_stats(void);
void gpu_reset_profile_stats(void);

#ifdef ENABLE_GLFW
// GLFW window and surface management
typedef void* GPUWindow;
typedef void* GPUSurface;

// Initialize GLFW (must be called before creating windows)
int gpu_glfw_init(GPUError* error);
void gpu_glfw_terminate(void);

// Window management
GPUWindow gpu_create_window(int width, int height, const char* title, GPUError* error);
void gpu_destroy_window(GPUWindow window);
int gpu_window_should_close(GPUWindow window);
void gpu_poll_events(void);
int gpu_window_get_key(GPUWindow window, int key);

// Surface management (for rendering to window)
GPUSurface gpu_create_surface_for_window(GPUContext ctx, GPUWindow window, GPUError* error);
void gpu_destroy_surface(GPUSurface surface);
void gpu_configure_surface(GPUSurface surface, int width, int height, GPUError* error);

// Get current texture from surface for rendering
typedef void* GPUTexture;
typedef void* GPUTextureView;
GPUTexture gpu_surface_get_current_texture(GPUSurface surface, GPUError* error);
GPUTextureView gpu_texture_create_view(GPUTexture texture, GPUError* error);
void gpu_texture_release(GPUTexture texture);
void gpu_texture_view_release(GPUTextureView view);

// Present the surface
void gpu_surface_present(GPUSurface surface);

// Command encoding for rendering
typedef void* GPUCommandEncoder;
typedef void* GPURenderPassEncoder;
typedef void* GPUCommandBuffer;
typedef void* GPURenderPipeline;

GPUCommandEncoder gpu_device_create_command_encoder(GPUContext ctx, GPUError* error);
GPURenderPassEncoder gpu_encoder_begin_render_pass(GPUCommandEncoder encoder, GPUTextureView view, GPUError* error);
void gpu_render_pass_set_pipeline(GPURenderPassEncoder pass, GPURenderPipeline pipeline);
void gpu_render_pass_draw(GPURenderPassEncoder pass, uint32_t vertex_count);
void gpu_render_pass_end(GPURenderPassEncoder pass);
GPUCommandBuffer gpu_encoder_finish(GPUCommandEncoder encoder, GPUError* error);
void gpu_queue_submit(GPUContext ctx, GPUCommandBuffer command, GPUError* error);
void gpu_command_buffer_release(GPUCommandBuffer command);
void gpu_command_encoder_release(GPUCommandEncoder encoder);
void gpu_render_pass_encoder_release(GPURenderPassEncoder pass);

// Pipeline creation
typedef void* GPUShaderModule;
GPUShaderModule gpu_create_shader_module(GPUContext ctx, const char* wgsl_code, GPUError* error);
GPURenderPipeline gpu_create_render_pipeline(GPUContext ctx, GPUShaderModule shader, int texture_format, GPUError* error);
void gpu_shader_module_release(GPUShaderModule shader);
void gpu_render_pipeline_release(GPURenderPipeline pipeline);

// Get texture format from surface
int gpu_surface_get_preferred_format(GPUSurface surface, GPUError* error);
#endif // ENABLE_GLFW

// Debug Ring Buffer API (for GPU printf)
typedef void* GPUDebugBuffer;

// Create a debug ring buffer for GPU printf-style debugging
// bufferSize: size in bytes (e.g., 64KB = 65536)
GPUDebugBuffer gpu_create_debug_buffer(GPUContext ctx, size_t bufferSize, GPUError* error);

// Destroy debug buffer
void gpu_destroy_debug_buffer(GPUDebugBuffer debugBuffer);

// Read debug buffer contents after kernel execution
// Returns number of entries read
uint32_t gpu_read_debug_buffer(GPUContext ctx, GPUDebugBuffer debugBuffer,
                                uint32_t* data, size_t maxEntries, GPUError* error);

// Clear debug buffer (reset write position)
void gpu_clear_debug_buffer(GPUContext ctx, GPUDebugBuffer debugBuffer);

#ifdef __cplusplus
}
#endif

#endif // GPU_WRAPPER_H
