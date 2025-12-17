{-# LANGUAGE CPP #-}

-- | Simple triangle rendering example using WebGPU and GLFW
-- This follows the pattern from https://developer.chrome.com/docs/web-platform/webgpu/build-app
module Main where

#ifdef ENABLE_GLFW

import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.GLFW
import Control.Monad (unless)
import Control.Exception (bracket, bracket_)

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
main = do
  putStrLn "Initializing WebGPU Triangle Demo..."

  -- Initialize GLFW
  glfwInit

  -- Create window and setup WebGPU
  bracket (createWindow 512 512 "WebGPU Triangle") destroyWindow $ \window -> do
    putStrLn "Window created"

    -- Create WebGPU context
    ctx <- createContext
    putStrLn "WebGPU context created"

    -- Create surface for the window
    bracket (createSurfaceForWindow ctx window) destroySurface $ \surface -> do
      putStrLn "Surface created"

      -- Get preferred format and configure surface
      format <- getSurfacePreferredFormat surface
      putStrLn $ "Surface format: " ++ show format
      configureSurface surface 512 512
      putStrLn "Surface configured"

      -- Create shader module
      bracket (createShaderModule ctx shaderCode) releaseShaderModule $ \shader -> do
        putStrLn "Shader compiled"

        -- Create render pipeline
        bracket (createRenderPipeline ctx shader format) releaseRenderPipeline $ \pipeline -> do
          putStrLn "Render pipeline created"
          putStrLn "Starting render loop..."

          -- Render loop
          renderLoop ctx window surface pipeline

  -- Clean up GLFW
  glfwTerminate
  putStrLn "Demo complete!"

-- Render loop
renderLoop :: Context -> Window -> Surface -> RenderPipeline -> IO ()
renderLoop ctx window surface pipeline = do
  shouldClose <- windowShouldClose window
  unless shouldClose $ do
    -- Get current texture from surface
    bracket (getCurrentTexture surface) releaseTexture $ \texture -> do

      -- Create texture view
      bracket (createTextureView texture) releaseTextureView $ \view -> do

        -- Create command encoder
        bracket (createCommandEncoder ctx) releaseCommandEncoder $ \encoder -> do

          -- Begin render pass
          bracket (beginRenderPass encoder view) releaseRenderPassEncoder $ \pass -> do
            -- Set pipeline and draw triangle
            setRenderPipeline pass pipeline
            draw pass 3  -- Draw 3 vertices (triangle)
            endRenderPass pass

          -- Finish encoding commands
          bracket (finishEncoder encoder) releaseCommandBuffer $ \commands -> do
            -- Submit commands to GPU
            submitCommand ctx commands

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
