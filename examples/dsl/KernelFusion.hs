{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-|
Kernel Fusion Example - Map Fusion

This demonstrates the new Kernel abstraction that enables kernel fusion.
Instead of multiple GPU kernel launches, we fuse operations into a single pass.

WITHOUT FUSION (3 separate kernels):
  Kernel 1: output1[i] = input[i] * 2.0
  Kernel 2: output2[i] = output1[i] + 1.0
  Kernel 3: output3[i] = relu(output2[i])

  Problem: 3 global memory roundtrips (slow!)

WITH FUSION (1 kernel):
  output[i] = relu(input[i] * 2.0 + 1.0)

  Benefit: Single pass, no intermediate storage needed!
-}

module Main where

import Prelude hiding (id, (.))
import qualified Prelude as P

import Graphics.WebGPU.Dawn hiding (Kernel)
import Graphics.WebGPU.Dawn.Tensor
import Graphics.WebGPU.Dawn.Types (AnyTensor(..))
import qualified Data.Vector.Storable as V
import Control.Exception (bracket)

import WGSL.DSL
import WGSL.Kernel (Kernel(..), mapK, (>>>))
import WGSL.Execute hiding (Kernel)

-- | ReLU activation function: max(0, x)
relu :: Exp F32 -> Exp F32
relu x = Max x (litF32 0.0)

-- | Load operation: reads from global memory at given index
--
-- Input: thread index (Exp I32)
-- Output: loaded value (Exp F32)
loadVec :: Ptr Storage (Array 256 F32) -> Kernel 256 1 1 (Exp I32) (Exp F32)
loadVec ptr = Kernel $ \idx -> readBuffer ptr idx

-- | Store operation: writes to global memory
--
-- Input: (index, value) tuple
-- Output: unit (side effect only)
storeVec :: Ptr Storage (Array 256 F32) -> Kernel 256 1 1 (Exp I32, Exp F32) ()
storeVec ptr = Kernel $ \(idx, val) -> writeBuffer ptr idx val

-- | Fused computation: multiply by 2, add 1, apply ReLU
--
-- This is the KEY demonstration of fusion:
-- Three operations composed into one with (>>>)
calcLogic :: Kernel 256 1 1 (Exp F32) (Exp F32)
calcLogic =
  mapK (* 2.0)     -- Step 1: multiply by 2
  >>> mapK (+ 1.0) -- Step 2: add 1
  >>> mapK relu    -- Step 3: apply ReLU

-- | Complete fused kernel shader
--
-- This builds the shader procedurally using the Kernel composition.
-- The result is a SINGLE shader that performs all operations in one pass.
fusedShaderDSL :: ShaderM ()
fusedShaderDSL = do
  -- Declare buffers with auto-binding
  input  <- declareInputBuffer "input" (TArray 256 TF32)
  output <- declareOutputBuffer "output" (TArray 256 TF32)

  -- Get thread ID
  gid <- globalId
  let idx = i32 (vecX gid)

  -- Execute the fused kernel!
  -- This looks like function composition but generates imperative code
  let pairWithIndex :: Kernel 256 1 1 (Exp F32) (Exp I32, Exp F32)
      pairWithIndex = Kernel $ \val -> return (idx, val)

      mainKernel :: Kernel 256 1 1 (Exp I32) ()
      mainKernel =
        loadVec input           -- Load input[idx]
        >>> calcLogic           -- Apply: (* 2) >>> (+ 1) >>> relu
        >>> pairWithIndex       -- Pair index with result
        >>> storeVec output     -- Store to output[idx]

  -- Run the kernel
  unKernel mainKernel idx

-- | Build shader module
buildFusionModule :: ShaderModule
buildFusionModule = buildShaderWithAutoBinding (256, 1, 1) fusedShaderDSL

-- | CPU reference implementation (for verification)
cpuFused :: V.Vector Float -> V.Vector Float
cpuFused input = V.map (\x -> P.max 0 (x P.* 2.0 P.+ 1.0)) input

main :: IO ()
main = do
  putStrLn "=== Kernel Fusion Example: Map Fusion ==="
  putStrLn ""
  putStrLn "Demonstrating fusion of: Load -> Multiply -> Add -> ReLU -> Store"
  putStrLn ""

  -- Generate shader
  let shaderModule = buildFusionModule
  putStrLn "Generated WGSL (fused single-pass shader):"
  putStrLn "==========================================="
  putStrLn P.$ generateWGSL shaderModule
  putStrLn ""

  putStrLn "Notice: The shader contains a SINGLE function that performs:"
  putStrLn "  1. Load from input buffer"
  putStrLn "  2. Multiply by 2.0"
  putStrLn "  3. Add 1.0"
  putStrLn "  4. Apply ReLU (max with 0)"
  putStrLn "  5. Store to output buffer"
  putStrLn ""
  putStrLn "All in ONE GPU kernel launch - no intermediate storage!"
  putStrLn ""

  -- Run on GPU
  bracket createContext destroyContext P.$ \ctx -> do
    putStrLn "Running on GPU..."

    -- Create test data: [-5, -4, ..., 4, 5, ...]
    let inputData = V.fromList [fromIntegral i P.- 128.0 :: Float | i <- [0..255]]
    let expectedOutput = cpuFused inputData

    -- Create tensors
    inputTensor <- createTensorWithDataTyped ctx (Shape [256]) inputData
    outputTensor <- createTensorTyped ctx (Shape [256]) :: IO (Tensor 'F32)

    -- Execute fused kernel
    executeShaderNamed ctx shaderModule
      [ ("input", AnyTensor inputTensor)
      , ("output", AnyTensor outputTensor)
      ]
      (WorkgroupSize 1 1 1)  -- 1 workgroup of 256 threads

    -- Read results
    gpuResult <- fromGPU ctx outputTensor 256 :: IO (V.Vector Float)

    -- Verify
    let maxDiff = V.maximum P.$ V.zipWith (\a b -> P.abs (a P.- b)) expectedOutput gpuResult

    putStrLn P.$ "✓ GPU execution completed"
    putStrLn P.$ "  Max difference from CPU: " P.++ show maxDiff
    putStrLn P.$ "  First 5 CPU results:  " P.++ show (V.take 5 expectedOutput)
    putStrLn P.$ "  First 5 GPU results:  " P.++ show (V.take 5 gpuResult)
    putStrLn ""

    if maxDiff P.< 0.001
      then putStrLn "✓✓✓ SUCCESS: Kernel fusion works correctly!"
      else putStrLn "✗✗✗ FAILURE: Results don't match"

    -- Cleanup
    destroyTensor inputTensor
    destroyTensor outputTensor

  putStrLn ""
  putStrLn "BENEFITS OF KERNEL FUSION:"
  putStrLn "=========================="
  putStrLn "• Single GPU kernel launch (not 3 separate launches)"
  putStrLn "• No intermediate global memory storage needed"
  putStrLn "• Better cache utilization"
  putStrLn "• Reduced memory bandwidth usage"
  putStrLn "• Composable with Category (>>>)"
  putStrLn ""
  putStrLn "This is the foundation for optimizing complex ML pipelines!"
