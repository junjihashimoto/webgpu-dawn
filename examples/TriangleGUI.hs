{-# LANGUAGE CPP #-}

-- | Triangle rendering example using ContT monad for resource management
-- This demonstrates automatic resource cleanup using the continuation monad
module Main where

#ifdef ENABLE_GLFW

import Graphics.WebGPU.Dawn.ContT
import Graphics.WebGPU.Dawn.GLFW
import Control.Monad (unless)

-- Shader code (WGSL) - renders a red triangle
shaderCode :: String
shaderCode = unlines
  [ "@vertex"
  , "fn vertexMain(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {"
  , "  const pos = array(vec2f(0, 1), vec2f(-1, -1), vec2f(1, -1));"
  , "  return vec4f(pos[i], 0, 1);"
  , "}"
  , ""
  , "@fragment"
  , "fn fragmentMain() -> @location(0) vec4f {"
  , "  return vec4f(1, 0, 0, 1);  // Red color"
  , "}"
  ]

main :: IO ()
main = evalContT $ do
  liftIO $ putStrLn "Initializing WebGPU Triangle Demo..."
  liftIO glfwInit

  -- Setup all resources using ContT for automatic cleanup
  window <- createWindow 512 512 "WebGPU Triangle"
  liftIO $ putStrLn "Window created"

  ctx <- createContext
  liftIO $ putStrLn "WebGPU context created"

  surface <- createSurfaceForWindow ctx window
  liftIO $ putStrLn "Surface created"

  -- Get preferred format and configure surface
  format <- liftIO $ getSurfacePreferredFormat surface
  liftIO $ putStrLn $ "Surface format: " ++ show format
  liftIO $ configureSurface surface 512 512
  liftIO $ putStrLn "Surface configured"

  shader <- createShaderModule ctx shaderCode
  liftIO $ putStrLn "Shader compiled"

  pipeline <- createRenderPipeline ctx shader format
  liftIO $ putStrLn "Render pipeline created"
  liftIO $ putStrLn "Starting render loop..."

  -- Render loop
  liftIO $ renderLoop ctx window surface pipeline

  -- Clean up GLFW
  liftIO glfwTerminate
  liftIO $ putStrLn "Demo complete!"

-- Render loop demonstrates nested ContT
-- Per-frame resources (texture, view, encoder, etc.) are created in inner scope
-- and automatically released after each frame, while persistent resources
-- (context, window, surface, pipeline) remain alive in outer scope
renderLoop :: Context -> Window -> Surface -> RenderPipeline -> IO ()
renderLoop ctx window surface pipeline = do
  shouldClose <- windowShouldClose window
  unless shouldClose $ do
    -- Inner scope: per-frame resources
    evalContT $ do
      texture <- createCurrentTexture surface
      view <- createTextureView texture
      encoder <- createCommandEncoder ctx

      pass <- createRenderPass encoder view
      liftIO $ do
        setRenderPipeline pass pipeline
        draw pass 3  -- Draw 3 vertices (triangle)
        endRenderPass pass

      commands <- createCommandBuffer encoder
      liftIO $ submitCommand ctx commands
    -- Per-frame resources automatically released here

    -- Present the frame
    surfacePresent surface

    -- Poll events
    pollEvents

    -- Continue loop
    renderLoop ctx window surface pipeline

#else

main :: IO ()
main = putStrLn "This example requires GLFW support. Please build with -fglfw flag."

#endif
