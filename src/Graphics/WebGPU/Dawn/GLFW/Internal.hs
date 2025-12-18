{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}

-- | Internal GLFW bindings - unsafe raw create/destroy functions
--
-- WARNING: This module exposes low-level resource management functions.
-- These functions do not automatically manage resource lifetimes and can
-- lead to resource leaks or use-after-free bugs if used incorrectly.
--
-- Users should prefer 'Graphics.WebGPU.Dawn.GLFW.ContT' which provides
-- safe automatic resource management.
module Graphics.WebGPU.Dawn.GLFW.Internal
  ( -- * GLFW Initialization
    glfwInit
  , glfwTerminate

    -- * Window Management
  , Window
  , createWindow
  , destroyWindow
  , windowShouldClose
  , pollEvents

    -- * Surface Management
  , Surface
  , createSurfaceForWindow
  , destroySurface
  , configureSurface
  , getSurfacePreferredFormat
  , surfacePresent

    -- * Texture and View Types
  , Texture
  , TextureView
  , getCurrentTexture
  , createTextureView
  , releaseTexture
  , releaseTextureView

    -- * Command Encoding
  , CommandEncoder
  , RenderPassEncoder
  , CommandBuffer
  , createCommandEncoder
  , beginRenderPass
  , setRenderPipeline
  , draw
  , endRenderPass
  , finishEncoder
  , submitCommand
  , releaseCommandBuffer
  , releaseCommandEncoder
  , releaseRenderPassEncoder

    -- * Pipeline Types
  , ShaderModule
  , RenderPipeline
  , createShaderModule
  , createRenderPipeline
  , releaseShaderModule
  , releaseRenderPipeline
  ) where

#ifdef ENABLE_GLFW

import Graphics.WebGPU.Dawn.Types (Context(..))
import Graphics.WebGPU.Dawn.Context (checkError)
import qualified Graphics.WebGPU.Dawn.Internal as I
import Foreign.Ptr
import Foreign.C.String
import Foreign.C.Types
import Foreign.Marshal.Alloc
import Foreign.Storable

-- | Opaque window type
newtype Window = Window (Ptr ())

-- | Opaque surface type
newtype Surface = Surface (Ptr ())

-- | Opaque texture type
newtype Texture = Texture (Ptr ())

-- | Opaque texture view type
newtype TextureView = TextureView (Ptr ())

-- | Opaque command encoder type
newtype CommandEncoder = CommandEncoder (Ptr ())

-- | Opaque render pass encoder type
newtype RenderPassEncoder = RenderPassEncoder (Ptr ())

-- | Opaque command buffer type
newtype CommandBuffer = CommandBuffer (Ptr ())

-- | Opaque shader module type
newtype ShaderModule = ShaderModule (Ptr ())

-- | Opaque render pipeline type
newtype RenderPipeline = RenderPipeline (Ptr ())

-- FFI imports
foreign import ccall "gpu_glfw_init"
  c_gpu_glfw_init :: Ptr I.GPUError -> IO CInt

foreign import ccall "gpu_glfw_terminate"
  c_gpu_glfw_terminate :: IO ()

foreign import ccall "gpu_create_window"
  c_gpu_create_window :: CInt -> CInt -> CString -> Ptr I.GPUError -> IO (Ptr ())

foreign import ccall "gpu_destroy_window"
  c_gpu_destroy_window :: Ptr () -> IO ()

foreign import ccall "gpu_window_should_close"
  c_gpu_window_should_close :: Ptr () -> IO CInt

foreign import ccall "gpu_poll_events"
  c_gpu_poll_events :: IO ()

foreign import ccall "gpu_create_surface_for_window"
  c_gpu_create_surface_for_window :: Ptr () -> Ptr () -> Ptr I.GPUError -> IO (Ptr ())

foreign import ccall "gpu_destroy_surface"
  c_gpu_destroy_surface :: Ptr () -> IO ()

foreign import ccall "gpu_configure_surface"
  c_gpu_configure_surface :: Ptr () -> CInt -> CInt -> Ptr I.GPUError -> IO ()

foreign import ccall "gpu_surface_get_preferred_format"
  c_gpu_surface_get_preferred_format :: Ptr () -> Ptr I.GPUError -> IO CInt

foreign import ccall "gpu_surface_get_current_texture"
  c_gpu_surface_get_current_texture :: Ptr () -> Ptr I.GPUError -> IO (Ptr ())

foreign import ccall "gpu_texture_create_view"
  c_gpu_texture_create_view :: Ptr () -> Ptr I.GPUError -> IO (Ptr ())

foreign import ccall "gpu_texture_release"
  c_gpu_texture_release :: Ptr () -> IO ()

foreign import ccall "gpu_texture_view_release"
  c_gpu_texture_view_release :: Ptr () -> IO ()

foreign import ccall "gpu_surface_present"
  c_gpu_surface_present :: Ptr () -> IO ()

foreign import ccall "gpu_device_create_command_encoder"
  c_gpu_device_create_command_encoder :: Ptr () -> Ptr I.GPUError -> IO (Ptr ())

foreign import ccall "gpu_encoder_begin_render_pass"
  c_gpu_encoder_begin_render_pass :: Ptr () -> Ptr () -> Ptr I.GPUError -> IO (Ptr ())

foreign import ccall "gpu_render_pass_set_pipeline"
  c_gpu_render_pass_set_pipeline :: Ptr () -> Ptr () -> IO ()

foreign import ccall "gpu_render_pass_draw"
  c_gpu_render_pass_draw :: Ptr () -> CUInt -> IO ()

foreign import ccall "gpu_render_pass_end"
  c_gpu_render_pass_end :: Ptr () -> IO ()

foreign import ccall "gpu_encoder_finish"
  c_gpu_encoder_finish :: Ptr () -> Ptr I.GPUError -> IO (Ptr ())

foreign import ccall "gpu_queue_submit"
  c_gpu_queue_submit :: Ptr () -> Ptr () -> Ptr I.GPUError -> IO ()

foreign import ccall "gpu_command_buffer_release"
  c_gpu_command_buffer_release :: Ptr () -> IO ()

foreign import ccall "gpu_command_encoder_release"
  c_gpu_command_encoder_release :: Ptr () -> IO ()

foreign import ccall "gpu_render_pass_encoder_release"
  c_gpu_render_pass_encoder_release :: Ptr () -> IO ()

foreign import ccall "gpu_create_shader_module"
  c_gpu_create_shader_module :: Ptr () -> CString -> Ptr I.GPUError -> IO (Ptr ())

foreign import ccall "gpu_create_render_pipeline"
  c_gpu_create_render_pipeline :: Ptr () -> Ptr () -> CInt -> Ptr I.GPUError -> IO (Ptr ())

foreign import ccall "gpu_shader_module_release"
  c_gpu_shader_module_release :: Ptr () -> IO ()

foreign import ccall "gpu_render_pipeline_release"
  c_gpu_render_pipeline_release :: Ptr () -> IO ()

-- High-level API

-- | Initialize GLFW. Must be called before creating windows.
glfwInit :: IO ()
glfwInit = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  result <- c_gpu_glfw_init errPtr
  checkError errPtr
  if result == 0
    then error "Failed to initialize GLFW"
    else return ()

-- | Terminate GLFW. Should be called when done with all windows.
glfwTerminate :: IO ()
glfwTerminate = c_gpu_glfw_terminate

-- | Create a window with the given dimensions and title
createWindow :: Int -> Int -> String -> IO Window
createWindow width height title = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  window <- withCString title $ \titlePtr ->
    c_gpu_create_window (fromIntegral width) (fromIntegral height) titlePtr errPtr
  checkError errPtr
  return $ Window window

-- | Destroy a window
destroyWindow :: Window -> IO ()
destroyWindow (Window win) = c_gpu_destroy_window win

-- | Check if a window should close
windowShouldClose :: Window -> IO Bool
windowShouldClose (Window win) = do
  result <- c_gpu_window_should_close win
  return $ result /= 0

-- | Poll for window events (must be called regularly in the render loop)
pollEvents :: IO ()
pollEvents = c_gpu_poll_events

-- | Create a WebGPU surface for the given window
createSurfaceForWindow :: Context -> Window -> IO Surface
createSurfaceForWindow ctx (Window win) = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  let (Context ctxPtr) = ctx
  surface <- c_gpu_create_surface_for_window (castPtr ctxPtr) win errPtr
  checkError errPtr
  return $ Surface surface

-- | Destroy a surface
destroySurface :: Surface -> IO ()
destroySurface (Surface surf) = c_gpu_destroy_surface surf

-- | Configure the surface with the given dimensions
configureSurface :: Surface -> Int -> Int -> IO ()
configureSurface (Surface surf) width height = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  c_gpu_configure_surface surf (fromIntegral width) (fromIntegral height) errPtr
  checkError errPtr

-- | Get the preferred texture format for the surface
getSurfacePreferredFormat :: Surface -> IO Int
getSurfacePreferredFormat (Surface surf) = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  format <- c_gpu_surface_get_preferred_format surf errPtr
  checkError errPtr
  return $ fromIntegral format

-- | Get the current texture from the surface for rendering
getCurrentTexture :: Surface -> IO Texture
getCurrentTexture (Surface surf) = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  texture <- c_gpu_surface_get_current_texture surf errPtr
  checkError errPtr
  return $ Texture texture

-- | Create a view of the texture for rendering
createTextureView :: Texture -> IO TextureView
createTextureView (Texture tex) = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  view <- c_gpu_texture_create_view tex errPtr
  checkError errPtr
  return $ TextureView view

-- | Release a texture
releaseTexture :: Texture -> IO ()
releaseTexture (Texture tex) = c_gpu_texture_release tex

-- | Release a texture view
releaseTextureView :: TextureView -> IO ()
releaseTextureView (TextureView view) = c_gpu_texture_view_release view

-- | Present the surface (swap buffers)
surfacePresent :: Surface -> IO ()
surfacePresent (Surface surf) = c_gpu_surface_present surf

-- | Create a command encoder for recording commands
createCommandEncoder :: Context -> IO CommandEncoder
createCommandEncoder ctx = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  let (Context ctxPtr) = ctx
  encoder <- c_gpu_device_create_command_encoder (castPtr ctxPtr) errPtr
  checkError errPtr
  return $ CommandEncoder encoder

-- | Begin a render pass with the given texture view
beginRenderPass :: CommandEncoder -> TextureView -> IO RenderPassEncoder
beginRenderPass (CommandEncoder enc) (TextureView view) = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  pass <- c_gpu_encoder_begin_render_pass enc view errPtr
  checkError errPtr
  return $ RenderPassEncoder pass

-- | Set the render pipeline for the render pass
setRenderPipeline :: RenderPassEncoder -> RenderPipeline -> IO ()
setRenderPipeline (RenderPassEncoder pass) (RenderPipeline pipeline) =
  c_gpu_render_pass_set_pipeline pass pipeline

-- | Draw vertices
draw :: RenderPassEncoder -> Int -> IO ()
draw (RenderPassEncoder pass) vertexCount =
  c_gpu_render_pass_draw pass (fromIntegral vertexCount)

-- | End the render pass
endRenderPass :: RenderPassEncoder -> IO ()
endRenderPass (RenderPassEncoder pass) = c_gpu_render_pass_end pass

-- | Finish recording commands and create a command buffer
finishEncoder :: CommandEncoder -> IO CommandBuffer
finishEncoder (CommandEncoder enc) = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  command <- c_gpu_encoder_finish enc errPtr
  checkError errPtr
  return $ CommandBuffer command

-- | Submit a command buffer to the GPU queue
submitCommand :: Context -> CommandBuffer -> IO ()
submitCommand ctx (CommandBuffer cmd) = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  let (Context ctxPtr) = ctx
  c_gpu_queue_submit (castPtr ctxPtr) cmd errPtr
  checkError errPtr

-- | Release a command buffer
releaseCommandBuffer :: CommandBuffer -> IO ()
releaseCommandBuffer (CommandBuffer cmd) = c_gpu_command_buffer_release cmd

-- | Release a command encoder
releaseCommandEncoder :: CommandEncoder -> IO ()
releaseCommandEncoder (CommandEncoder enc) = c_gpu_command_encoder_release enc

-- | Release a render pass encoder
releaseRenderPassEncoder :: RenderPassEncoder -> IO ()
releaseRenderPassEncoder (RenderPassEncoder pass) = c_gpu_render_pass_encoder_release pass

-- | Create a shader module from WGSL code
createShaderModule :: Context -> String -> IO ShaderModule
createShaderModule ctx code = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  let (Context ctxPtr) = ctx
  shader <- withCString code $ \codePtr ->
    c_gpu_create_shader_module (castPtr ctxPtr) codePtr errPtr
  checkError errPtr
  return $ ShaderModule shader

-- | Create a render pipeline from a shader module
createRenderPipeline :: Context -> ShaderModule -> Int -> IO RenderPipeline
createRenderPipeline ctx (ShaderModule shader) format = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  let (Context ctxPtr) = ctx
  pipeline <- c_gpu_create_render_pipeline (castPtr ctxPtr) shader (fromIntegral format) errPtr
  checkError errPtr
  return $ RenderPipeline pipeline

-- | Release a shader module
releaseShaderModule :: ShaderModule -> IO ()
releaseShaderModule (ShaderModule shader) = c_gpu_shader_module_release shader

-- | Release a render pipeline
releaseRenderPipeline :: RenderPipeline -> IO ()
releaseRenderPipeline (RenderPipeline pipeline) = c_gpu_render_pipeline_release pipeline

#endif
