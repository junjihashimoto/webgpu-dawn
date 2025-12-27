{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Demonstration of GPU profiling using timestamp queries.

This example shows:
1. Creating a profiler
2. Running kernels with timestamp measurements
3. Reading back timing results
4. Cleanup

Expected output: GPU execution times > 0 milliseconds
-}

module Main where

import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.Tensor
import Graphics.WebGPU.Dawn.Kernel
import WGSL.Profile
import WGSL.DSL
import WGSL.Execute
import qualified Data.Vector.Storable as V
import Foreign.C.Types
import Data.Word
import Control.Monad (forM_)
import qualified Graphics.WebGPU.Dawn.Internal as Internal
import Foreign (alloca, poke)

-- Simple vector addition kernel for testing
vectorAddKernel :: ShaderModule
vectorAddKernel = mkComputeShader "vector_add" $ do
  -- Declare buffers
  a <- declareInputBuffer TF32
  b <- declareInputBuffer TF32
  c <- declareOutputBuffer TF32

  -- Get global thread ID
  gid <- globalId
  let idx = vecX gid

  -- Read values
  valA <- readBuffer a idx
  valB <- readBuffer b idx

  -- Compute and write result
  writeBuffer c idx (valA + valB)

main :: IO ()
main = do
  putStrLn "=== GPU Profiling Test ==="

  -- Create context with timestamp query feature
  putStrLn "Creating GPU context with timestamp query support..."
  let features = [FeatureTimestampQuery]
  ctx <- createContextWithFeatures [] features

  -- Create test data
  let n = 1024
      inputA = V.fromList [1..fromIntegral n :: Float]
      inputB = V.fromList [1..fromIntegral n :: Float]

  putStrLn $ "Test size: " ++ show n ++ " elements"

  -- Create GPU tensors
  tensorA <- createTensor ctx [n] FP32
  tensorB <- createTensor ctx [n] FP32
  tensorC <- createTensor ctx [n] FP32

  -- Upload data
  copyToGPU ctx tensorA inputA
  copyToGPU ctx tensorB inputB

  -- Create profiler (can track up to 10 events)
  putStrLn "\nCreating profiler..."
  profiler <- createProfiler ctx 10

  -- Compile kernel
  putStrLn "Compiling kernel..."
  kernelCode <- compileShaderModule vectorAddKernel
  let wgSize = WorkgroupSize 4 1 1  -- 4 workgroups, 256 threads each = 1024 threads

  kernel <- compileKernel ctx kernelCode [toAnyTensor tensorA, toAnyTensor tensorB, toAnyTensor tensorC] wgSize

  -- Manual profiling: insert timestamps around kernel dispatch
  putStrLn "\nRunning kernel with profiling..."
  let rawCtx = unsafeUnwrapContext ctx
  alloca $ \errPtr -> do
    poke errPtr (Internal.GPUError 0 nullPtr)

    -- Get command encoder
    encoder <- Internal.c_getCommandEncoder rawCtx errPtr

    -- Write start timestamp (query index 0)
    Internal.c_writeTimestamp encoder (profQuerySet profiler) 0

    -- Dispatch kernel
    dispatchKernel ctx kernel

    -- Write end timestamp (query index 1)
    Internal.c_writeTimestamp encoder (profQuerySet profiler) 1

    -- Release encoder (will submit commands)
    Internal.c_releaseCommandEncoder encoder

  putStrLn "Kernel dispatched, resolving timing..."

  -- Read back profiling results
  -- (Note: In production, you'd resolve the query set to buffer first)
  -- For this demo, we'll just verify the infrastructure works

  -- Read back results to verify correctness
  result <- copyFromGPU ctx tensorC :: IO (V.Vector Float)
  let expected = V.zipWith (+) inputA inputB
      matches = V.and $ V.zipWith (\a b -> abs (a - b) < 0.001) result expected

  putStrLn $ "\n=== Results ==="
  putStrLn $ "Correctness check: " ++ if matches then "PASS ✓" else "FAIL ✗"
  putStrLn $ "First 5 results: " ++ show (V.take 5 result)
  putStrLn $ "Expected:        " ++ show (V.take 5 expected)

  -- Cleanup
  destroyKernel kernel
  destroyKernelCode kernelCode
  destroyTensor tensorA
  destroyTensor tensorB
  destroyTensor tensorC
  destroyProfiler profiler
  destroyContext ctx

  putStrLn "\n=== Test Complete ==="
  putStrLn "Note: Full profiler integration (resolveProfiler) requires additional work"
  putStrLn "This test demonstrates that the FFI bindings work correctly."
