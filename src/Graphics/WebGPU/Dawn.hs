{-|
Module      : Graphics.WebGPU.Dawn
Description : High-level Haskell bindings to WebGPU Dawn
Copyright   : (c) 2025
License     : MIT
Maintainer  : junji.hashimoto@gmail.com

This module provides high-level, type-safe Haskell bindings to Google's Dawn
WebGPU implementation. It enables GPU computing and graphics programming from
Haskell with automatic resource management.

= Quick Start

Here's a simple example of GPU-accelerated vector addition:

@
import Graphics.WebGPU.Dawn
import qualified Data.Vector.Storable as V

main :: IO ()
main = withContext $ \\ctx -> do
  -- Create input vectors
  let a = V.fromList [1, 2, 3, 4] :: V.Vector Float
      b = V.fromList [5, 6, 7, 8] :: V.Vector Float
      shape = Shape [4]

  -- Create GPU tensors
  tensorA <- createTensorWithData ctx shape a
  tensorB <- createTensorWithData ctx shape b
  tensorC <- createTensor ctx shape F32

  -- Compile and run kernel
  let shader = unlines
        [ "\@group(0) \@binding(0) var<storage, read> a: array<f32>;"
        , "\@group(0) \@binding(1) var<storage, read> b: array<f32>;"
        , "\@group(0) \@binding(2) var<storage, read_write> c: array<f32>;"
        , ""
        , "\@compute \@workgroup_size(256)"
        , "fn main(\@builtin(global_invocation_id) gid: vec3<u32>) {"
        , "  c[gid.x] = a[gid.x] + b[gid.x];"
        , "}"
        ]

  code <- createKernelCode shader
  kernel <- compileKernel ctx code [tensorA, tensorB, tensorC]
                         (WorkgroupSize 1 1 1)
  dispatchKernel ctx kernel

  -- Read results
  result <- fromGPU ctx tensorC 4
  print result  -- [6.0, 8.0, 10.0, 12.0]
@
-}

module Graphics.WebGPU.Dawn
  ( -- * Context Management
    Context
  , withContext
  , createContext
  , destroyContext

    -- * Tensor Operations
  , Tensor
  , Shape(..)
  , NumType(..)
  , createTensor
  , createTensorWithData
  , destroyTensor
  , TensorData(..)

    -- * Data Transfer
  , toGPU
  , fromGPU

    -- * Kernel Management
  , Kernel
  , KernelCode
  , createKernelCode
  , setWorkgroupSize
  , setEntryPoint
  , destroyKernelCode
  , compileKernel
  , destroyKernel
  , dispatchKernel

    -- * Configuration
  , WorkgroupSize(..)
  , defaultWorkgroupSize

    -- * Error Handling
  , GPUException(..)

    -- * Utilities
  , numTypeSize
  , shapeSize
  , shapeDims
  ) where

import Graphics.WebGPU.Dawn.Types
import Graphics.WebGPU.Dawn.Context
import Graphics.WebGPU.Dawn.Tensor
import Graphics.WebGPU.Dawn.Kernel
