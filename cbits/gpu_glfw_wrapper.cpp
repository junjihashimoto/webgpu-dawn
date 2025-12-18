#ifdef ENABLE_GLFW

#include "gpu_wrapper.h"
#include <gpu.hpp>
#include <webgpu/webgpu_cpp.h>
#include <webgpu/webgpu_glfw.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <cstring>

// Error handling macros
#define CLEAR_ERROR(err) if (err) { (err)->code = 0; (err)->message = nullptr; }
#define SET_ERROR(err, c, msg) if (err) { (err)->code = c; (err)->message = strdup(msg); }

// Context implementation (shared with gpu_cpp_bridge.cpp)
struct GPUContextImpl {
    gpu::Context* ctx;
};

// Surface implementation
struct GPUSurfaceImpl {
    wgpu::Surface surface;
    gpu::Context* ctx;
    int width;
    int height;
    wgpu::TextureFormat format;
};

// Initialize GLFW
int gpu_glfw_init(GPUError* error) {
    try {
        CLEAR_ERROR(error);
        if (!glfwInit()) {
            SET_ERROR(error, 1, "Failed to initialize GLFW");
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return 0;
    }
}

void gpu_glfw_terminate(void) {
    glfwTerminate();
}

// Window management
GPUWindow gpu_create_window(int width, int height, const char* title, GPUError* error) {
    try {
        CLEAR_ERROR(error);

        // Tell GLFW not to create an OpenGL context
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!window) {
            SET_ERROR(error, 1, "Failed to create GLFW window");
            return nullptr;
        }

        return static_cast<GPUWindow>(window);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

void gpu_destroy_window(GPUWindow window) {
    if (window) {
        glfwDestroyWindow(static_cast<GLFWwindow*>(window));
    }
}

int gpu_window_should_close(GPUWindow window) {
    if (!window) return 1;
    return glfwWindowShouldClose(static_cast<GLFWwindow*>(window));
}

void gpu_poll_events(void) {
    glfwPollEvents();
}

int gpu_window_get_key(GPUWindow window, int key) {
    if (!window) {
        return GLFW_RELEASE;
    }
    auto glfw_window = static_cast<GLFWwindow*>(window);
    return glfwGetKey(glfw_window, key);
}

// Surface management
GPUSurface gpu_create_surface_for_window(GPUContext ctx, GPUWindow window, GPUError* error) {
    try {
        CLEAR_ERROR(error);

        if (!ctx || !window) {
            SET_ERROR(error, 3, "Invalid context or window");
            return nullptr;
        }

        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);
        auto glfw_window = static_cast<GLFWwindow*>(window);

        // Create surface using the GLFW helper
        wgpu::Instance instance(ctx_impl->ctx->instance);
        wgpu::Surface surface = wgpu::glfw::CreateSurfaceForWindow(instance, glfw_window);

        if (!surface) {
            SET_ERROR(error, 1, "Failed to create surface");
            return nullptr;
        }

        // Get window size
        int width, height;
        glfwGetFramebufferSize(glfw_window, &width, &height);

        auto impl = new GPUSurfaceImpl();
        impl->surface = surface;
        impl->ctx = ctx_impl->ctx;
        impl->width = width;
        impl->height = height;

        return static_cast<GPUSurface>(impl);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

void gpu_destroy_surface(GPUSurface surface) {
    if (surface) {
        auto impl = static_cast<GPUSurfaceImpl*>(surface);
        delete impl;
    }
}

int gpu_surface_get_preferred_format(GPUSurface surface, GPUError* error) {
    try {
        CLEAR_ERROR(error);

        if (!surface) {
            SET_ERROR(error, 3, "Invalid surface");
            return 0;
        }

        auto impl = static_cast<GPUSurfaceImpl*>(surface);

        // Get preferred format
        wgpu::SurfaceCapabilities capabilities;
        wgpu::Adapter adapter(impl->ctx->adapter);
        impl->surface.GetCapabilities(adapter, &capabilities);

        if (capabilities.formatCount > 0) {
            impl->format = capabilities.formats[0];
            return static_cast<int>(impl->format);
        }

        // Default to BGRA8Unorm
        impl->format = wgpu::TextureFormat::BGRA8Unorm;
        return static_cast<int>(impl->format);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return 0;
    }
}

void gpu_configure_surface(GPUSurface surface, int width, int height, GPUError* error) {
    try {
        CLEAR_ERROR(error);

        if (!surface) {
            SET_ERROR(error, 3, "Invalid surface");
            return;
        }

        auto impl = static_cast<GPUSurfaceImpl*>(surface);
        impl->width = width;
        impl->height = height;

        // Configure the surface
        wgpu::SurfaceConfiguration config{};
        wgpu::Device device(impl->ctx->device);
        config.device = device;
        config.format = impl->format;
        config.width = width;
        config.height = height;
        config.usage = wgpu::TextureUsage::RenderAttachment;
        config.presentMode = wgpu::PresentMode::Fifo;

        impl->surface.Configure(&config);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
    }
}

GPUTexture gpu_surface_get_current_texture(GPUSurface surface, GPUError* error) {
    try {
        CLEAR_ERROR(error);

        if (!surface) {
            SET_ERROR(error, 3, "Invalid surface");
            return nullptr;
        }

        auto impl = static_cast<GPUSurfaceImpl*>(surface);

        wgpu::SurfaceTexture surfaceTexture;
        impl->surface.GetCurrentTexture(&surfaceTexture);

        // Return the texture (caller must release)
        return new wgpu::Texture(surfaceTexture.texture);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

GPUTextureView gpu_texture_create_view(GPUTexture texture, GPUError* error) {
    try {
        CLEAR_ERROR(error);

        if (!texture) {
            SET_ERROR(error, 3, "Invalid texture");
            return nullptr;
        }

        auto tex = static_cast<wgpu::Texture*>(texture);
        wgpu::TextureView view = tex->CreateView();

        return new wgpu::TextureView(view);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

void gpu_texture_release(GPUTexture texture) {
    if (texture) {
        delete static_cast<wgpu::Texture*>(texture);
    }
}

void gpu_texture_view_release(GPUTextureView view) {
    if (view) {
        delete static_cast<wgpu::TextureView*>(view);
    }
}

void gpu_surface_present(GPUSurface surface) {
    if (surface) {
        auto impl = static_cast<GPUSurfaceImpl*>(surface);
        impl->surface.Present();
    }
}

// Command encoding for rendering
GPUCommandEncoder gpu_device_create_command_encoder(GPUContext ctx, GPUError* error) {
    try {
        CLEAR_ERROR(error);

        if (!ctx) {
            SET_ERROR(error, 3, "Invalid context");
            return nullptr;
        }

        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);
        wgpu::Device device(ctx_impl->ctx->device);
        wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

        return new wgpu::CommandEncoder(encoder);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

GPURenderPassEncoder gpu_encoder_begin_render_pass(GPUCommandEncoder encoder, GPUTextureView view, GPUError* error) {
    try {
        CLEAR_ERROR(error);

        if (!encoder || !view) {
            SET_ERROR(error, 3, "Invalid encoder or view");
            return nullptr;
        }

        auto enc = static_cast<wgpu::CommandEncoder*>(encoder);
        auto tex_view = static_cast<wgpu::TextureView*>(view);

        wgpu::RenderPassColorAttachment attachment{};
        attachment.view = *tex_view;
        attachment.loadOp = wgpu::LoadOp::Clear;
        attachment.storeOp = wgpu::StoreOp::Store;
        attachment.clearValue = {0.0, 0.0, 0.0, 1.0}; // Black background

        wgpu::RenderPassDescriptor renderpass{};
        renderpass.colorAttachmentCount = 1;
        renderpass.colorAttachments = &attachment;

        wgpu::RenderPassEncoder pass = enc->BeginRenderPass(&renderpass);

        return new wgpu::RenderPassEncoder(pass);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

void gpu_render_pass_set_pipeline(GPURenderPassEncoder pass, GPURenderPipeline pipeline) {
    if (pass && pipeline) {
        auto p = static_cast<wgpu::RenderPassEncoder*>(pass);
        auto pip = static_cast<wgpu::RenderPipeline*>(pipeline);
        p->SetPipeline(*pip);
    }
}

void gpu_render_pass_draw(GPURenderPassEncoder pass, uint32_t vertex_count) {
    if (pass) {
        auto p = static_cast<wgpu::RenderPassEncoder*>(pass);
        p->Draw(vertex_count);
    }
}

void gpu_render_pass_end(GPURenderPassEncoder pass) {
    if (pass) {
        auto p = static_cast<wgpu::RenderPassEncoder*>(pass);
        p->End();
    }
}

GPUCommandBuffer gpu_encoder_finish(GPUCommandEncoder encoder, GPUError* error) {
    try {
        CLEAR_ERROR(error);

        if (!encoder) {
            SET_ERROR(error, 3, "Invalid encoder");
            return nullptr;
        }

        auto enc = static_cast<wgpu::CommandEncoder*>(encoder);
        wgpu::CommandBuffer commands = enc->Finish();

        return new wgpu::CommandBuffer(commands);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

void gpu_queue_submit(GPUContext ctx, GPUCommandBuffer command, GPUError* error) {
    try {
        CLEAR_ERROR(error);

        if (!ctx || !command) {
            SET_ERROR(error, 3, "Invalid context or command");
            return;
        }

        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);
        auto cmd = static_cast<wgpu::CommandBuffer*>(command);

        wgpu::Device device(ctx_impl->ctx->device);
        device.GetQueue().Submit(1, cmd);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
    }
}

void gpu_command_buffer_release(GPUCommandBuffer command) {
    if (command) {
        delete static_cast<wgpu::CommandBuffer*>(command);
    }
}

void gpu_command_encoder_release(GPUCommandEncoder encoder) {
    if (encoder) {
        delete static_cast<wgpu::CommandEncoder*>(encoder);
    }
}

void gpu_render_pass_encoder_release(GPURenderPassEncoder pass) {
    if (pass) {
        delete static_cast<wgpu::RenderPassEncoder*>(pass);
    }
}

// Shader and pipeline creation
GPUShaderModule gpu_create_shader_module(GPUContext ctx, const char* wgsl_code, GPUError* error) {
    try {
        CLEAR_ERROR(error);

        if (!ctx || !wgsl_code) {
            SET_ERROR(error, 3, "Invalid context or shader code");
            return nullptr;
        }

        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);

        wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
        wgslDesc.code = wgsl_code;

        wgpu::ShaderModuleDescriptor desc{};
        desc.nextInChain = &wgslDesc;

        wgpu::Device device(ctx_impl->ctx->device);
        wgpu::ShaderModule shader = device.CreateShaderModule(&desc);

        return new wgpu::ShaderModule(shader);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

GPURenderPipeline gpu_create_render_pipeline(GPUContext ctx, GPUShaderModule shader, int texture_format, GPUError* error) {
    try {
        CLEAR_ERROR(error);

        if (!ctx || !shader) {
            SET_ERROR(error, 3, "Invalid context or shader");
            return nullptr;
        }

        auto ctx_impl = static_cast<GPUContextImpl*>(ctx);
        auto shd = static_cast<wgpu::ShaderModule*>(shader);

        wgpu::ColorTargetState colorTarget{};
        colorTarget.format = static_cast<wgpu::TextureFormat>(texture_format);

        wgpu::FragmentState fragment{};
        fragment.module = *shd;
        fragment.entryPoint = "fragmentMain";
        fragment.targetCount = 1;
        fragment.targets = &colorTarget;

        wgpu::RenderPipelineDescriptor pipelineDesc{};
        pipelineDesc.vertex.module = *shd;
        pipelineDesc.vertex.entryPoint = "vertexMain";
        pipelineDesc.fragment = &fragment;
        pipelineDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;

        wgpu::Device device(ctx_impl->ctx->device);
        wgpu::RenderPipeline pipeline = device.CreateRenderPipeline(&pipelineDesc);

        return new wgpu::RenderPipeline(pipeline);
    } catch (const std::exception& e) {
        SET_ERROR(error, 1, e.what());
        return nullptr;
    }
}

void gpu_shader_module_release(GPUShaderModule shader) {
    if (shader) {
        delete static_cast<wgpu::ShaderModule*>(shader);
    }
}

void gpu_render_pipeline_release(GPURenderPipeline pipeline) {
    if (pipeline) {
        delete static_cast<wgpu::RenderPipeline*>(pipeline);
    }
}

#endif // ENABLE_GLFW
