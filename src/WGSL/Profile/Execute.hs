{-# LANGUAGE ScopedTypeVariables #-}

{-|
Module      : WGSL.Profile.Execute
Description : Profiling-aware shader execution
Copyright   : (c) 2025
License     : MIT

High-level API for executing shaders with GPU profiling enabled.
Wraps kernel dispatches with timestamp queries for performance measurement.
-}

module WGSL.Profile.Execute
  ( -- * Profiled execution
    executeShaderWithProfiling
  , ProfiledExecution(..)

    -- * Re-exports
  , module WGSL.Profile
  ) where

import WGSL.Profile
import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.Types (AnyTensor(..))
import Graphics.WebGPU.Dawn.Tensor
import Graphics.WebGPU.Dawn.Kernel
import qualified Graphics.WebGPU.Dawn.Internal as Internal
import WGSL.Execute (compileShaderModule)
import WGSL.AST (ShaderModule)
import Foreign
import Foreign.C.Types
import Data.Word

-- | Result of profiled shader execution
data ProfiledExecution = ProfiledExecution
  { profileEvents :: ![ProfileEvent]
  , profileTotalMs :: !Double
  } deriving (Show, Eq)

-- | Execute a shader with profiling enabled
--
-- This function:
-- 1. Compiles the shader
-- 2. Inserts timestamp queries before/after kernel dispatch
-- 3. Resolves timing results
-- 4. Returns profiling data
--
-- Example:
-- @
--   result <- executeShaderWithProfiling ctx "VectorAdd" myShader tensors wgSize
--   putStrLn $ "GPU time: " ++ show (profileTotalMs result) ++ " ms"
-- @
executeShaderWithProfiling
  :: Context             -- ^ GPU context
  -> String              -- ^ Event name for profiling
  -> ShaderModule        -- ^ Shader to execute
  -> [Tensor dtype]      -- ^ Input/output tensors
  -> WorkgroupSize       -- ^ Dispatch configuration
  -> IO ProfiledExecution
executeShaderWithProfiling ctx eventName shaderModule tensors wgSize = do
  let rawCtx = unsafeUnwrapContext ctx

  -- Create profiler (1 event = 2 timestamps)
  profiler <- createProfiler rawCtx 1

  -- Compile shader and kernel
  kernelCode <- compileShaderModule shaderModule

  -- Convert tensors to AnyTensor
  let anyTensors = map AnyTensor tensors

  -- Compile kernel
  kernel <- compileKernelHeterogeneous ctx kernelCode anyTensors wgSize

  -- Execute with timestamps
  alloca $ \errPtr -> do
    poke errPtr (Internal.GPUError 0 nullPtr)

    -- Get command encoder
    encoder <- Internal.c_getCommandEncoder rawCtx errPtr

    -- Write start timestamp
    Internal.c_writeTimestamp encoder (profQuerySet profiler) 0

    -- Dispatch kernel
    dispatchKernel ctx kernel

    -- Write end timestamp
    Internal.c_writeTimestamp encoder (profQuerySet profiler) 1

    -- Resolve query set to buffer
    Internal.c_resolveQuerySet encoder (profQuerySet profiler) 0 2
                      (profResolveBuffer profiler) 0

    -- Release encoder (submits commands)
    Internal.c_releaseCommandEncoder encoder

  -- Wait for GPU to finish
  waitAll ctx

  -- Read timing results
  events <- resolveProfiler profiler

  -- Cleanup
  destroyKernel kernel
  destroyKernelCode kernelCode
  destroyProfiler profiler

  let totalMs = if null events
                then 0.0
                else sum $ map eventDurationMs events

  return $ ProfiledExecution events totalMs
