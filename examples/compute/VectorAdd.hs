{-|
Example: Vector Addition on GPU

This example demonstrates basic GPU compute by adding two vectors element-wise.
It shows:
- Creating GPU tensors
- Uploading data to GPU
- Compiling and running WGSL compute shaders
- Reading results back to CPU
-}

module Main where

import Graphics.WebGPU.Dawn
import qualified Data.Vector.Storable as V
import Text.Printf

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
main = do
  putStrLn "=== GPU Vector Addition Example ==="
  putStrLn ""

  -- Create test vectors
  let n = 1024
      vecA = V.generate n (\i -> fromIntegral i) :: V.Vector Float
      vecB = V.generate n (\i -> fromIntegral (n - i)) :: V.Vector Float
      shape = Shape [n]

  putStrLn $ "Vector size: " ++ show n
  putStrLn $ "First 5 elements of A: " ++ show (V.take 5 vecA)
  putStrLn $ "First 5 elements of B: " ++ show (V.take 5 vecB)
  putStrLn ""

  -- Run on GPU
  withContext $ \ctx -> do
    putStrLn "Creating GPU tensors..."
    tensorA <- createTensorWithData ctx shape vecA
    tensorB <- createTensorWithData ctx shape vecB
    tensorC <- createTensor ctx shape F32

    putStrLn "Compiling shader..."
    code <- createKernelCode vectorAddShader

    putStrLn "Creating kernel..."
    let numWorkgroups = (n + 255) `div` 256  -- Ceiling division
    kernel <- compileKernel ctx code [tensorA, tensorB, tensorC]
                           (WorkgroupSize numWorkgroups 1 1)

    putStrLn "Dispatching kernel..."
    dispatchKernel ctx kernel

    putStrLn "Reading results from GPU..."
    result <- fromGPU ctx tensorC n

    putStrLn ""
    putStrLn $ "First 5 elements of C: " ++ show (V.take 5 result)
    putStrLn $ "Last 5 elements of C: " ++ show (V.drop (n - 5) result)

    -- Verify correctness
    let expected = V.zipWith (+) vecA vecB
        correct = result == expected

    putStrLn ""
    if correct
      then putStrLn "✓ Results correct!"
      else putStrLn "✗ Results incorrect!"

    -- Cleanup
    destroyTensor tensorA
    destroyTensor tensorB
    destroyTensor tensorC
    destroyKernel kernel
    destroyKernelCode code

  putStrLn ""
  putStrLn "Example completed successfully."
