{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Parallel reduction using shared memory and barriers
-- This example demonstrates Phase 3 features: workgroup memory and synchronization
module Main where

import Prelude
import qualified Data.Vector.Storable as V
import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.Tensor
import Graphics.WebGPU.Dawn.Kernel
import Control.Exception (bracket)

-- | Parallel reduction shader using shared memory
-- Sums 256 elements using shared memory and barriers
reductionWGSL :: String
reductionWGSL = unlines
  [ "@group(0) @binding(0)"
  , "var<storage, read> input: array<f32>;"
  , ""
  , "@group(0) @binding(1)"
  , "var<storage, read_write> output: array<f32>;"
  , ""
  , "// Shared memory for parallel reduction"
  , "var<workgroup> shared_data: array<f32, 256>;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(local_invocation_id) local_id: vec3<u32>,"
  , "        @builtin(global_invocation_id) global_id: vec3<u32>) {"
  , "    let tid = local_id.x;"
  , "    let gid = global_id.x;"
  , ""
  , "    // Load data into shared memory"
  , "    shared_data[tid] = input[gid];"
  , "    workgroupBarrier();"
  , ""
  , "    // Parallel reduction in shared memory"
  , "    var stride = 128u;"
  , "    while (stride > 0u) {"
  , "        if (tid < stride) {"
  , "            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];"
  , "        }"
  , "        workgroupBarrier();"
  , "        stride = stride / 2u;"
  , "    }"
  , ""
  , "    // Thread 0 writes the result"
  , "    if (tid == 0u) {"
  , "        output[0] = shared_data[0];"
  , "    }"
  , "}"
  ]

main :: IO ()
main = do
  putStrLn "=== Phase 3: Shared Memory & Barriers ==="
  putStrLn "Parallel Reduction Example"
  putStrLn ""

  bracket createContext destroyContext $ \ctx -> do
    putStrLn "✓ GPU context initialized"

    -- Create input: [1.0, 2.0, 3.0, ..., 256.0]
    let inputData = V.fromList [fromIntegral i :: Float | i <- [1..256]]
        expectedSum = V.sum inputData  -- Should be 32896.0 = 256 * 257 / 2

    putStrLn $ "✓ Input: 256 elements [1.0 .. 256.0]"
    putStrLn $ "✓ Expected sum: " ++ show expectedSum

    -- Create tensors
    inputTensor <- createTensorWithDataTyped ctx (Shape [256]) inputData
    outputTensor <- createTensorTyped ctx (Shape [1]) :: IO (Tensor 'F32)
    putStrLn "✓ Tensors created"

    -- Compile and run shader
    kernelCode <- createKernelCode reductionWGSL
    kernel <- compileKernel ctx kernelCode [inputTensor, outputTensor] (WorkgroupSize 1 1 1)
    putStrLn "✓ Shader compiled (using shared memory)"

    dispatchKernel ctx kernel
    putStrLn "✓ Parallel reduction executed"

    -- Read result
    result <- fromGPU ctx outputTensor 0 :: IO (V.Vector Float)

    putStrLn ""
    putStrLn $ "Result vector length: " ++ show (V.length result)
    putStrLn $ "Result vector: " ++ show result

    if V.null result
      then putStrLn "\n✗✗✗ FAILURE: No result returned from GPU"
      else do
        let actualSum = V.head result
        putStrLn $ "Result: " ++ show actualSum
        putStrLn $ "Expected: " ++ show expectedSum
        putStrLn $ "Difference: " ++ show (abs (actualSum - expectedSum))

        if abs (actualSum - expectedSum) < 0.01
          then putStrLn "\n✓✓✓ SUCCESS: Parallel reduction works correctly!"
          else putStrLn "\n✗✗✗ FAILURE: Result doesn't match"

    -- Clean up
    destroyKernel kernel
    destroyKernelCode kernelCode
    destroyTensor inputTensor
    destroyTensor outputTensor

  putStrLn ""
  putStrLn "Phase 3 Features Demonstrated:"
  putStrLn "  ✓ Shared memory (var<workgroup>)"
  putStrLn "  ✓ Synchronization barriers (workgroupBarrier)"
  putStrLn "  ✓ Parallel reduction algorithm"
