{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Integration layer between WGSL DSL and webgpu-dawn kernel execution
-- This module bridges the gap between shader code generation and GPU execution
module WGSL.Execute
  ( -- * High-level execution
    executeShader
  , executeShaderAsync
  , executeShaderWith
  , executeShaderNamed  -- NEW: Safe named execution

    -- * Lower-level control
  , compileShaderModule
  , createKernelFromDSL

    -- * Re-exports for convenience
  , module Graphics.WebGPU.Dawn
  , module Graphics.WebGPU.Dawn.Tensor
  , module Graphics.WebGPU.Dawn.Kernel
  ) where

import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.Tensor
import Graphics.WebGPU.Dawn.Kernel
import Graphics.WebGPU.Dawn.Types
import WGSL.AST
import WGSL.CodeGen
import Data.List (sortOn)
import qualified Data.Map as Map
import Control.Exception (throwIO)

-- | Compile a shader module into a KernelCode object
-- This generates WGSL source from the DSL and creates kernel code
compileShaderModule :: ShaderModule -> IO KernelCode
compileShaderModule shaderModule = do
  let wgslSource = generateWGSL shaderModule
  createKernelCode wgslSource

-- | Create a compiled kernel from a shader module with bound tensors
-- This is a convenience wrapper that:
-- 1. Generates WGSL code from the shader module
-- 2. Creates kernel code
-- 3. Compiles the kernel with the provided tensors and workgroup configuration
createKernelFromDSL
  :: Context
  -> ShaderModule
  -> [Tensor dtype]       -- ^ Tensors to bind (must match shader's @binding declarations)
  -> WorkgroupSize        -- ^ Number of workgroups to dispatch
  -> IO Kernel
createKernelFromDSL ctx shaderModule tensors wgSize = do
  kernelCode <- compileShaderModule shaderModule
  kernel <- compileKernel ctx kernelCode tensors wgSize
  destroyKernelCode kernelCode  -- Clean up intermediate kernel code
  return kernel

-- | Execute a shader synchronously with the given tensors
-- This is the simplest way to run a WGSL DSL shader:
-- 1. Generates WGSL code
-- 2. Compiles the kernel
-- 3. Executes it synchronously
-- 4. Cleans up the kernel
--
-- Example:
-- @
--   executeShader ctx myShader [inputTensor, outputTensor] (WorkgroupSize 256 1 1)
-- @
executeShader
  :: Context
  -> ShaderModule
  -> [Tensor dtype]
  -> WorkgroupSize
  -> IO ()
executeShader ctx shaderModule tensors wgSize = do
  kernel <- createKernelFromDSL ctx shaderModule tensors wgSize
  dispatchKernel ctx kernel
  destroyKernel kernel

-- | Execute a shader asynchronously (non-blocking)
-- The kernel executes in the background. Remember to call 'waitAll' before
-- reading results from GPU tensors.
--
-- This allows GPU pipelining for better performance when running multiple kernels.
executeShaderAsync
  :: Context
  -> ShaderModule
  -> [Tensor dtype]
  -> WorkgroupSize
  -> IO ()
executeShaderAsync ctx shaderModule tensors wgSize = do
  kernel <- createKernelFromDSL ctx shaderModule tensors wgSize
  dispatchKernelAsync ctx kernel
  destroyKernel kernel

-- | Execute a shader with custom kernel configuration
-- This provides more control over the kernel execution, allowing you to:
-- - Set a custom entry point (default is "main")
-- - Reuse a compiled kernel for multiple dispatches
-- - Customize workgroup size
executeShaderWith
  :: Context
  -> ShaderModule
  -> String              -- ^ Entry point function name
  -> [Tensor dtype]
  -> WorkgroupSize
  -> IO ()
executeShaderWith ctx shaderModule entryPoint tensors wgSize = do
  kernelCode <- compileShaderModule shaderModule
  setEntryPoint kernelCode entryPoint
  kernel <- compileKernel ctx kernelCode tensors wgSize
  destroyKernelCode kernelCode
  dispatchKernel ctx kernel
  destroyKernel kernel

-- | Execute a shader with NAMED buffer bindings (SAFE!)
-- This is the RECOMMENDED way to execute shaders as it prevents binding mismatch errors.
--
-- Unlike executeShader which requires tensors in list order (error-prone),
-- this function matches tensors to buffers by name using the moduleBindings metadata.
--
-- Example:
-- @
--   executeShaderNamed ctx myShader
--     [ ("inputA", AnyTensor tensorA)   -- F32 tensor
--     , ("inputB", AnyTensor tensorB)   -- I32 tensor
--     , ("output", AnyTensor tensorOut) -- U32 tensor
--     ] (WorkgroupSize 256 1 1)
-- @
--
-- Features:
-- - Type-safe: Allows mixing different tensor types (F32, I32, U32, etc.)
-- - Name-based: Match buffers by name, not fragile list order
-- - Validated: Checks that all required buffers are provided
executeShaderNamed
  :: Context
  -> ShaderModule
  -> [(String, AnyTensor)]  -- ^ Named tensors
  -> WorkgroupSize
  -> IO ()
executeShaderNamed ctx shaderModule namedTensors wgSize = do
  -- Build a map from names to tensors
  let tensorMap = Map.fromList namedTensors
      bindings = moduleBindings shaderModule

  -- Check that all required buffers are provided
  let requiredNames = map fst bindings
      providedNames = map fst namedTensors
      missingNames = filter (`notElem` providedNames) requiredNames
      extraNames = filter (`notElem` requiredNames) providedNames

  -- Validate
  if not (null missingNames)
    then throwIO $ InvalidDataSize $ "Missing buffers: " ++ show missingNames
    else if not (null extraNames)
    then throwIO $ InvalidDataSize $ "Unknown buffers: " ++ show extraNames
    else do
      -- Sort tensors by binding index
      let sortedTensors = map snd $ sortOn fst
            [ (bindingIdx, tensorMap Map.! name)
            | (name, bindingIdx) <- bindings
            ]

      -- Compile and execute
      kernelCode <- compileShaderModule shaderModule
      kernel <- compileKernelHeterogeneous ctx kernelCode sortedTensors wgSize
      destroyKernelCode kernelCode
      dispatchKernel ctx kernel
      destroyKernel kernel
