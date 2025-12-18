{-# LANGUAGE CPP #-}

-- | ContT-based helpers for GLFW resource management
-- This module provides cleaner APIs for using GLFW with ContT monad
module Graphics.WebGPU.Dawn.GLFW.ContT
  ( -- * GLFW Initialization
    glfwInit
  , glfwTerminate
    -- * Window Management
  , createWindow
  , Window
  , windowShouldClose
  , windowGetKey
  , pollEvents
    -- * Surface Management
  , createSurfaceForWindow
  , Surface
  , getSurfacePreferredFormat
  , configureSurface
  , surfacePresent
    -- * Texture and View Management
  , createCurrentTexture
  , createTextureView
  , Texture
  , TextureView
    -- * Command Encoding
  , createCommandEncoder
  , createRenderPass
  , createCommandBuffer
  , CommandEncoder
  , RenderPassEncoder
  , CommandBuffer
  , setRenderPipeline
  , draw
  , endRenderPass
  , submitCommand
    -- * Pipeline Management
  , createShaderModule
  , createRenderPipeline
  , ShaderModule
  , RenderPipeline
  ) where

#ifdef ENABLE_GLFW

import Control.Monad.Trans.Cont (ContT(..))
import Control.Exception (bracket)
import Graphics.WebGPU.Dawn.Types
import Graphics.WebGPU.Dawn.GLFW.Internal
  ( Window
  , Surface
  , Texture
  , TextureView
  , CommandEncoder
  , RenderPassEncoder
  , CommandBuffer
  , ShaderModule
  , RenderPipeline
  , windowShouldClose
  , windowGetKey
  , pollEvents
  , surfacePresent
  , getSurfacePreferredFormat
  , configureSurface
  , setRenderPipeline
  , draw
  , endRenderPass
  , submitCommand
  )
import qualified Graphics.WebGPU.Dawn.GLFW.Internal as GLFW

-- | Initialize GLFW (re-exported from GLFW module)
glfwInit :: IO ()
glfwInit = GLFW.glfwInit

-- | Terminate GLFW (re-exported from GLFW module)
glfwTerminate :: IO ()
glfwTerminate = GLFW.glfwTerminate

-- | Create a window with automatic cleanup
createWindow :: Int -> Int -> String -> ContT r IO Window
createWindow width height title = ContT $ \k ->
  bracket (GLFW.createWindow width height title) GLFW.destroyWindow k

-- | Create a surface for a window with automatic cleanup
createSurfaceForWindow :: Context -> Window -> ContT r IO Surface
createSurfaceForWindow ctx win = ContT $ \k ->
  bracket (GLFW.createSurfaceForWindow ctx win) GLFW.destroySurface k

-- | Get current texture from surface with automatic cleanup
createCurrentTexture :: Surface -> ContT r IO Texture
createCurrentTexture surface = ContT $ \k ->
  bracket (GLFW.getCurrentTexture surface) GLFW.releaseTexture k

-- | Create a texture view with automatic cleanup
createTextureView :: Texture -> ContT r IO TextureView
createTextureView texture = ContT $ \k ->
  bracket (GLFW.createTextureView texture) GLFW.releaseTextureView k

-- | Create a command encoder with automatic cleanup
createCommandEncoder :: Context -> ContT r IO CommandEncoder
createCommandEncoder ctx = ContT $ \k ->
  bracket (GLFW.createCommandEncoder ctx) GLFW.releaseCommandEncoder k

-- | Begin a render pass with automatic cleanup
createRenderPass :: CommandEncoder -> TextureView -> ContT r IO RenderPassEncoder
createRenderPass encoder view = ContT $ \k ->
  bracket (GLFW.beginRenderPass encoder view) GLFW.releaseRenderPassEncoder k

-- | Create a command buffer from encoder with automatic cleanup
createCommandBuffer :: CommandEncoder -> ContT r IO CommandBuffer
createCommandBuffer encoder = ContT $ \k ->
  bracket (GLFW.finishEncoder encoder) GLFW.releaseCommandBuffer k

-- | Create a shader module with automatic cleanup
createShaderModule :: Context -> String -> ContT r IO ShaderModule
createShaderModule ctx code = ContT $ \k ->
  bracket (GLFW.createShaderModule ctx code) GLFW.releaseShaderModule k

-- | Create a render pipeline with automatic cleanup
createRenderPipeline :: Context -> ShaderModule -> Int -> ContT r IO RenderPipeline
createRenderPipeline ctx shader format = ContT $ \k ->
  bracket (GLFW.createRenderPipeline ctx shader format) GLFW.releaseRenderPipeline k

#else

-- Stub implementations when GLFW is not enabled
glfwInit :: IO ()
glfwInit = error "GLFW support not enabled"

glfwTerminate :: IO ()
glfwTerminate = error "GLFW support not enabled"

#endif
