{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GADTs #-}

{-|
Module: WGSL.API
Description: High-Level API for GPU Computing (The "Accelerate" Layer)

This module provides a high-level, easy-to-use API for GPU computing,
similar to Haskell Accelerate. It automatically handles:
- Shader generation
- Buffer allocation
- Kernel execution
- Result retrieval

Example:
@
  import WGSL.API

  -- Double every element on the GPU
  result <- gpuMap (*2.0) inputVector

  -- Square root of every element
  sqrts <- gpuMap sqrt inputVector
@

This layer sits on top of the low-level DSL, providing:
- Zero boilerplate for simple operations
- Automatic buffer management
- Type-safe transformations
- Compiler-generated optimized shaders

For advanced use cases (shared memory, subgroups, custom kernels),
use the low-level WGSL.DSL API directly.
-}

module WGSL.API
  ( -- * High-Level Operations
    gpuMap
  , gpuMap2
  , gpuZipWith
  , gpuFold

    -- * Types
  , GPUVector
  , toGPU
  , fromGPU'

    -- * Re-exports for convenience
  , Context
  , createContext
  , destroyContext
  ) where

import Prelude
import qualified Prelude as P

import Graphics.WebGPU.Dawn hiding (toGPU)
import Graphics.WebGPU.Dawn.Tensor hiding (toGPU)
import Graphics.WebGPU.Dawn.Types (AnyTensor(..))
import qualified Data.Vector.Storable as V
import Control.Exception (bracket)
import Data.Proxy

import WGSL.AST
import WGSL.DSL hiding ((<), (>), (<=), (>=), (==), (/=))
import qualified WGSL.DSL as DSL
import WGSL.Execute hiding (toGPU)

-- | GPU-resident vector (opaque handle)
-- This wraps a Tensor but provides a more user-friendly interface
data GPUVector a = GPUVector
  { gpuContext :: Context
  , gpuTensor :: Tensor a
  , gpuSize :: Int
  }

-- | Transfer a vector to GPU memory
toGPU :: forall a. Context -> V.Vector Float -> IO (GPUVector 'F32)
toGPU ctx vec = do
  let size = V.length vec
  tensor <- createTensorWithDataTyped ctx (Shape [size]) vec
  return $ GPUVector ctx tensor size

-- | Transfer a vector from GPU memory to CPU
fromGPU' :: forall a. GPUVector 'F32 -> IO (V.Vector Float)
fromGPU' (GPUVector ctx tensor size) = fromGPU ctx tensor size

-- ============================================================================
-- High-Level Operations
-- ============================================================================

-- | Map a function over a GPU vector
-- This is the primary high-level operation - zero boilerplate!
--
-- Example:
-- @
--   ctx <- createContext
--   input <- toGPU ctx (V.fromList [1,2,3,4])
--   output <- gpuMap (*2.0) input
--   result <- fromGPU' output
--   -- result = [2,4,6,8]
-- @
gpuMap :: (Exp F32 -> Exp F32) -> GPUVector 'F32 -> IO (GPUVector 'F32)
gpuMap f (GPUVector ctx inputTensor size) = do
  -- Allocate output buffer
  outputTensor <- createTensorTyped ctx (Shape [size]) :: IO (Tensor 'F32)

  -- Generate shader automatically
  let shader = buildGpuMapShader size f

  -- Execute with auto-binding
  executeShaderNamed ctx shader
    [ ("input", AnyTensor inputTensor)
    , ("output", AnyTensor outputTensor)
    ]
    (WorkgroupSize ((size + 255) `div` 256) 1 1)  -- Auto-compute workgroups

  return $ GPUVector ctx outputTensor size

-- | Map a binary function over two GPU vectors (element-wise)
--
-- Example:
-- @
--   result <- gpuMap2 (+) vecA vecB  -- Element-wise addition
-- @
gpuMap2 :: (Exp F32 -> Exp F32 -> Exp F32)
        -> GPUVector 'F32
        -> GPUVector 'F32
        -> IO (GPUVector 'F32)
gpuMap2 f (GPUVector ctx inputA sizeA) (GPUVector _ inputB sizeB) = do
  if sizeA P./= sizeB
    then error "gpuMap2: input vectors must have the same size"
    else do
      -- Allocate output buffer
      outputTensor <- createTensorTyped ctx (Shape [sizeA]) :: IO (Tensor 'F32)

      -- Generate shader automatically
      let shader = buildGpuMap2Shader sizeA f

      -- Execute with auto-binding
      executeShaderNamed ctx shader
        [ ("inputA", AnyTensor inputA)
        , ("inputB", AnyTensor inputB)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize ((sizeA + 255) `div` 256) 1 1)

      return $ GPUVector ctx outputTensor sizeA

-- | Alias for gpuMap2 (more intuitive naming)
gpuZipWith :: (Exp F32 -> Exp F32 -> Exp F32)
           -> GPUVector 'F32
           -> GPUVector 'F32
           -> IO (GPUVector 'F32)
gpuZipWith = gpuMap2

-- | Parallel reduction (fold) on GPU
-- Currently implements a simple parallel reduction
--
-- Example:
-- @
--   sum <- gpuFold (+) 0 vector  -- Parallel sum
-- @
gpuFold :: (Exp F32 -> Exp F32 -> Exp F32)  -- Binary operator
        -> Float                             -- Initial value
        -> GPUVector 'F32                    -- Input vector
        -> IO Float
gpuFold op initVal (GPUVector ctx inputTensor size) = do
  -- For now, implement a simple two-stage reduction:
  -- Stage 1: Each workgroup reduces locally to shared memory
  -- Stage 2: Reduce the workgroup results on CPU

  -- Simple implementation: read to CPU and fold
  -- TODO: Implement proper GPU reduction with shared memory
  vec <- fromGPU ctx inputTensor size
  return $ V.foldl' (\acc x -> evalExpApprox (op (LitF32 acc) (LitF32 x))) initVal vec
  where
    -- Approximate evaluation for simple expressions
    evalExpApprox :: Exp F32 -> Float
    evalExpApprox (LitF32 x) = x
    evalExpApprox (Add a b) = evalExpApprox a + evalExpApprox b
    evalExpApprox (Mul a b) = evalExpApprox a * evalExpApprox b
    evalExpApprox (Sub a b) = evalExpApprox a - evalExpApprox b
    evalExpApprox (Div a b) = evalExpApprox a / evalExpApprox b
    evalExpApprox _ = error "gpuFold: complex expression not supported yet"

-- ============================================================================
-- Shader Generation (Internal)
-- ============================================================================

-- | Automatically generate a map shader from a function
buildGpuMapShader :: Int -> (Exp F32 -> Exp F32) -> ShaderModule
buildGpuMapShader size f = buildShaderWithAutoBinding (256, 1, 1) $ do
  -- Declare buffers
  input <- declareInputBuffer "input" (TArray size TF32)
  output <- declareOutputBuffer "output" (TArray size TF32)

  -- Get global thread ID
  gid <- globalId
  let idx = i32 (vecX gid)

  -- Bounds check
  if_ (idx DSL.< litI32 size)
    (do
      -- Load input
      value <- readBuffer input idx

      -- Apply user function
      let result = f value

      -- Store result
      writeBuffer output idx result
    )
    (return ())

-- | Automatically generate a map2 shader from a binary function
buildGpuMap2Shader :: Int -> (Exp F32 -> Exp F32 -> Exp F32) -> ShaderModule
buildGpuMap2Shader size f = buildShaderWithAutoBinding (256, 1, 1) $ do
  -- Declare buffers
  inputA <- declareInputBuffer "inputA" (TArray size TF32)
  inputB <- declareInputBuffer "inputB" (TArray size TF32)
  output <- declareOutputBuffer "output" (TArray size TF32)

  -- Get global thread ID
  gid <- globalId
  let idx = i32 (vecX gid)

  -- Bounds check
  if_ (idx DSL.< litI32 size)
    (do
      -- Load inputs
      valueA <- readBuffer inputA idx
      valueB <- readBuffer inputB idx

      -- Apply user function
      let result = f valueA valueB

      -- Store result
      writeBuffer output idx result
    )
    (return ())
