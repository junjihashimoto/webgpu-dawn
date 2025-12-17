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

// Numeric types enum matching gpu.hpp
typedef enum {
    GPU_F16 = 0,
    GPU_F32 = 1,
    GPU_F64 = 2,
    GPU_I8  = 3,
    GPU_I16 = 4,
    GPU_I32 = 5,
    GPU_I64 = 6,
    GPU_U8  = 7,
    GPU_U16 = 8,
    GPU_U32 = 9,
    GPU_U64 = 10,
    GPU_UNKNOWN = 11
} GPUNumType;

// Error handling
typedef struct {
    int code;
    const char* message;
} GPUError;

// Context management
GPUContext gpu_create_context(GPUError* error);
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
                            GPUError* error);
void gpu_destroy_kernel(GPUKernel kernel);
void gpu_dispatch_kernel(GPUContext ctx, GPUKernel kernel, GPUError* error);

// Utility functions
size_t gpu_size_of_type(GPUNumType dtype);
const char* gpu_type_name(GPUNumType dtype);

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

#ifdef __cplusplus
}
#endif

#endif // GPU_WRAPPER_H
