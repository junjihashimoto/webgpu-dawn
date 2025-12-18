{-|
Module      : Graphics.WebGPU.Dawn
Description : High-level Haskell bindings to WebGPU Dawn
Copyright   : (c) 2025
License     : MIT
Maintainer  : junji.hashimoto@gmail.com

This module provides high-level, type-safe Haskell bindings to Google's Dawn
WebGPU implementation. It enables GPU computing and graphics programming from
Haskell with automatic resource management.

= Recommended Usage

For safe automatic resource management, use the ContT-based APIs:

* For GPU compute: 'Graphics.WebGPU.Dawn.ContT'
* For graphics with GLFW: 'Graphics.WebGPU.Dawn.GLFW'
* For traditional bracket-style: 'Graphics.WebGPU.Dawn.IO'

Both ContT and GLFW modules provide automatic resource cleanup.

= Quick Start (ContT Style - Recommended)

@
import Graphics.WebGPU.Dawn.ContT
import Control.Monad.Trans.Cont (evalContT)
import qualified Data.Vector.Storable as V

main :: IO ()
main = evalContT $ do
  ctx <- createContext

  -- Create GPU tensors with automatic cleanup
  let a = V.fromList [1, 2, 3, 4] :: V.Vector Float
      b = V.fromList [5, 6, 7, 8] :: V.Vector Float
      shape = Shape [4]

  tensorA <- createTensorWithData ctx shape a
  tensorB <- createTensorWithData ctx shape b
  tensorC <- createTensor ctx shape F32

  -- Compile and run kernel
  code <- createKernelCode shaderSource
  kernel <- createKernel ctx code [tensorA, tensorB, tensorC]
                        (WorkgroupSize 1 1 1)
  liftIO $ dispatchKernel ctx kernel

  -- Read results
  result <- liftIO $ fromGPU ctx tensorC 4
  liftIO $ print result  -- [6.0, 8.0, 10.0, 12.0]
  -- All resources automatically cleaned up here!
@

= Warning About Low-Level APIs

This module exports low-level @create*@ and @destroy*@ functions for
compatibility. Using these directly without proper cleanup can lead to
resource leaks. Prefer the safe APIs mentioned above.
-}

module Graphics.WebGPU.Dawn
  ( -- * Context Management
    Context
  , withContext
  , withContextFeatures
  , createContext
  , createContextWithFeatures
  , destroyContext
  , WGPUFeatureName(..)

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
