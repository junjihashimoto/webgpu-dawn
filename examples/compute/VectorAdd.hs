{-|
Example: Vector Addition on GPU with ContT

This example demonstrates basic GPU compute by adding two vectors element-wise.
It shows:
- Creating GPU tensors with automatic cleanup
- Uploading data to GPU
- Compiling and running WGSL compute shaders
- Reading results back to CPU
- Automatic resource management with ContT
-}

module Main where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V

-- WGSL compute shader for vector addition
vectorAddShader :: String
vectorAddShader = unlines
  [ "@group(0) @binding(0) var<storage, read_write> a: array<f32>;"
  , "@group(0) @binding(1) var<storage, read_write> b: array<f32>;"
  , "@group(0) @binding(2) var<storage, read_write> c: array<f32>;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let i = gid.x;"
  , "  if (i < arrayLength(&a)) {"
  , "    c[i] = a[i] + b[i];"
  , "  }"
  , "}"
  ]

main :: IO ()
main = evalContT $ do
  liftIO $ putStrLn "=== GPU Vector Addition Example ==="
  liftIO $ putStrLn ""

  -- Create test vectors
  let n = 1024
      vecA = V.generate n (\i -> fromIntegral i) :: V.Vector Float
      vecB = V.generate n (\i -> fromIntegral (n - i)) :: V.Vector Float
      shape = Shape [n]

  liftIO $ putStrLn $ "Vector size: " ++ show n
  liftIO $ putStrLn $ "First 5 elements of A: " ++ show (V.take 5 vecA)
  liftIO $ putStrLn $ "First 5 elements of B: " ++ show (V.take 5 vecB)
  liftIO $ putStrLn ""

  -- Run on GPU with automatic resource management
  ctx <- createContext
  liftIO $ putStrLn "Creating GPU tensors..."
  tensorA <- createTensorWithData ctx shape vecA
  tensorB <- createTensorWithData ctx shape vecB
  tensorC <- createTensor ctx shape F32

  liftIO $ putStrLn "Compiling shader..."
  code <- createKernelCode vectorAddShader

  liftIO $ putStrLn "Creating kernel..."
  let numWorkgroups = (n + 255) `div` 256  -- Ceiling division
  kernel <- createKernel ctx code [tensorA, tensorB, tensorC]
                        (WorkgroupSize numWorkgroups 1 1)

  liftIO $ putStrLn "Dispatching kernel..."
  liftIO $ dispatchKernel ctx kernel

  liftIO $ putStrLn "Reading results from GPU..."
  result <- liftIO $ fromGPU ctx tensorC n

  liftIO $ putStrLn ""
  liftIO $ putStrLn $ "First 5 elements of C: " ++ show (V.take 5 result)
  liftIO $ putStrLn $ "Last 5 elements of C: " ++ show (V.drop (n - 5) result)

  -- Verify correctness
  let expected = V.zipWith (+) vecA vecB
      correct = result == expected

  liftIO $ putStrLn ""
  if correct
    then liftIO $ putStrLn "✓ Results correct!"
    else liftIO $ putStrLn "✗ Results incorrect!"

  -- All resources automatically cleaned up here!
  liftIO $ putStrLn ""
  liftIO $ putStrLn "Example completed successfully."
