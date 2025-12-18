{-# LANGUAGE ScopedTypeVariables #-}

module Graphics.WebGPU.Dawn.IO
  ( -- * Context Management
    withContext
  , withContextFeatures
    -- * Tensor Management
  , withTensor
  , withTensorWithData
    -- * Kernel Management
  , withKernelCode
  , withKernel
    -- * Re-exports from base modules
  , module Graphics.WebGPU.Dawn.Types
  , TensorData
  , WorkgroupSize(..)
  , defaultWorkgroupSize
  ) where

import Control.Exception (bracket)
import Data.Vector.Storable (Vector)
import Graphics.WebGPU.Dawn.Types
import qualified Graphics.WebGPU.Dawn.Context as C
import Graphics.WebGPU.Dawn.Context (WGPUFeatureName)
import qualified Graphics.WebGPU.Dawn.Tensor as T
import Graphics.WebGPU.Dawn.Tensor (TensorData)
import qualified Graphics.WebGPU.Dawn.Kernel as K
import Graphics.WebGPU.Dawn.Kernel (WorkgroupSize(..), defaultWorkgroupSize)

-- | Create a context with automatic cleanup
withContext :: (Context -> IO a) -> IO a
withContext = bracket C.createContext C.destroyContext

-- | Create a context with features and automatic cleanup
withContextFeatures :: [String] -> [WGPUFeatureName] -> (Context -> IO a) -> IO a
withContextFeatures toggles features =
  bracket (C.createContextWithFeatures toggles features) C.destroyContext

-- | Create a tensor with automatic cleanup
withTensor :: Context -> Shape -> NumType -> (Tensor -> IO a) -> IO a
withTensor ctx shape dtype =
  bracket (T.createTensor ctx shape dtype) T.destroyTensor

-- | Create a tensor with data and automatic cleanup
withTensorWithData :: TensorData a => Context -> Shape -> Vector a -> (Tensor -> IO b) -> IO b
withTensorWithData ctx shape vec =
  bracket (T.createTensorWithData ctx shape vec) T.destroyTensor

-- | Create kernel code with automatic cleanup
withKernelCode :: String -> (KernelCode -> IO a) -> IO a
withKernelCode wgslSource =
  bracket (K.createKernelCode wgslSource) K.destroyKernelCode

-- | Compile a kernel with automatic cleanup
withKernel :: Context -> KernelCode -> [Tensor] -> WorkgroupSize -> (Kernel -> IO a) -> IO a
withKernel ctx code tensors wgSize =
  bracket (K.compileKernel ctx code tensors wgSize) K.destroyKernel
