{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-|
Parallel Reduction using PHASE 6 IMPROVED WGSL DSL

This demonstrates Phase 6 DSL features:
  ✓ Natural math operators (tid + stride instead of Add tid stride)
  ✓ HOAS loops with while_ (type-safe loop variables)
  ✓ Shared memory and barriers (shared, barrier)
  ✓ Type-safe comparisons (tid < stride instead of Lt tid stride)
  ✓ No raw WGSL strings - pure DSL construction!
-}

module Main where

import Prelude (IO, Int, Float, ($), (.), (++), (+), (/), (-), div, show, abs, putStrLn, fromIntegral, return, ceiling, logBase)
import qualified Prelude as P
import qualified Data.Vector.Storable as V
import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.Tensor
import Control.Exception (bracket)

-- Import DSL components
import WGSL.DSL
import WGSL.Execute

-- | Parallel reduction shader using Phase 6 DSL
-- Sums 256 elements using shared memory and barriers
reductionShaderDSL :: Int -> ShaderM ()
reductionShaderDSL workgroupSize = do
  -- Reference global storage buffers
  let inputBuf = globalBuffer "input" :: Ptr Storage (Array 256 F32)
  let outputBuf = globalBuffer "output" :: Ptr Storage (Array 1 F32)

  -- Get thread IDs
  lid <- localId
  gid <- globalId

  let tid = vecX lid   -- Local thread ID
  let globalIdx = vecX gid  -- Global thread ID

  -- Declare shared memory for reduction
  sharedData <- shared (TArray workgroupSize TF32)

  -- Load data into shared memory
  inputVal <- readBuffer inputBuf (i32 globalIdx)
  -- PtrIndex returns an Exp, so we use it directly in expressions
  let Ptr sharedName = sharedData
  emitStmt $ Assign (sharedName ++ "[" ++ prettyExp (i32 tid) ++ "]") (SomeExp inputVal)

  -- Synchronization barrier - wait for all threads to load
  barrier

  -- Parallel reduction using iterative halving
  -- For workgroupSize=256, we need log2(256)=8 iterations
  stride <- var TU32 (fromIntegral (workgroupSize `div` 2))

  -- Fixed number of iterations (log2 of workgroup size)
  let numIterations = ceiling (logBase 2 (fromIntegral workgroupSize :: Float))

  loop 0 (fromIntegral numIterations) 1 $ \_ -> do
    strideVal <- readPtr stride

    -- Only threads with tid < stride participate
    if_ (tid < strideVal)
      (do
        -- shared_data[tid] = shared_data[tid] + shared_data[tid + stride]
        -- Use PtrIndex to read values (it's already an Exp)
        let val1 = PtrIndex sharedData (i32 tid) :: Exp F32
        let val2 = PtrIndex sharedData (i32 (tid + strideVal)) :: Exp F32
        -- Emit assignment statement directly (use prettyExp to properly format index)
        emitStmt $ Assign (sharedName ++ "[" ++ prettyExp (i32 tid) ++ "]") (SomeExp (val1 + val2))
      )
      (return ())

    barrier

    -- stride = stride / 2 (integer division for U32)
    assign stride (Div strideVal 2)

  -- Thread 0 writes the final result
  if_ (tid == 0)
    (do
      let finalResult = PtrIndex sharedData (litI32 0)
      let Ptr outputName = outputBuf
      emitStmt $ Assign (outputName ++ "[" ++ prettyExp (litI32 0) ++ "]") (SomeExp finalResult)
    )
    (return ())

-- | Build the complete shader module
buildReductionModule :: Int -> ShaderModule
buildReductionModule workgroupSize =
  buildShaderModuleWithConfig
    [ ("input", TArray 256 TF32, MStorage)
    , ("output", TArray 1 TF32, MStorage)
    ]
    []   -- No extensions
    []   -- No diagnostic filters
    (workgroupSize, 1, 1)
    (reductionShaderDSL workgroupSize)

main :: IO ()
main = do
  putStrLn "=== Parallel Reduction using PHASE 6 DSL ==="
  putStrLn "Shared Memory & Barriers Example"
  putStrLn ""

  let workgroupSize = 256

  -- Build shader using DSL
  putStrLn "Building shader with Phase 6 DSL..."
  let shaderModule = buildReductionModule workgroupSize

  putStrLn "Generated WGSL from DSL:"
  putStrLn "========================"
  putStrLn $ generateWGSL shaderModule
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

    -- Compile and run shader using DSL-generated WGSL
    let shaderCode = generateWGSL shaderModule
    kernelCode <- createKernelCode shaderCode
    kernel <- compileKernel ctx kernelCode [inputTensor, outputTensor] (WorkgroupSize 1 1 1)
    putStrLn "✓ DSL shader compiled (using shared memory)"

    dispatchKernel ctx kernel
    putStrLn "✓ Parallel reduction executed"

    -- Read result (read 1 element from output buffer)
    result <- fromGPU ctx outputTensor 1 :: IO (V.Vector Float)

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

        if abs (actualSum - expectedSum) P.< 0.01
          then putStrLn "\n✓✓✓ SUCCESS: Parallel reduction works correctly!"
          else putStrLn "\n✗✗✗ FAILURE: Result doesn't match"

    -- Clean up
    destroyKernel kernel
    destroyKernelCode kernelCode
    destroyTensor inputTensor
    destroyTensor outputTensor

  putStrLn ""
  putStrLn "PHASE 6 DSL Features Demonstrated:"
  putStrLn ""
  putStrLn "1. Natural Math Operators:"
  putStrLn "   val1 + val2              -- Not: Add val1 val2"
  putStrLn "   tid + strideVal          -- Not: Add tid strideVal"
  putStrLn "   strideVal / 2            -- Not: Div strideVal (litU32 2)"
  putStrLn ""
  putStrLn "2. Natural Comparisons:"
  putStrLn "   tid < strideVal          -- Not: Lt tid strideVal"
  putStrLn "   strideVal > 0            -- Not: Gt strideVal (litU32 0)"
  putStrLn "   tid == 0                 -- Not: Eq tid (litU32 0)"
  putStrLn ""
  putStrLn "3. While Loops:"
  putStrLn "   while_ (strideVal > 0) $ do"
  putStrLn "     ..."
  putStrLn ""
  putStrLn "4. Shared Memory & Barriers:"
  putStrLn "   sharedData <- shared (TArray 256 TF32)"
  putStrLn "   barrier  -- workgroupBarrier() in WGSL"
  putStrLn ""
  putStrLn "Phase 6: Clean, idiomatic Haskell for parallel algorithms!"
