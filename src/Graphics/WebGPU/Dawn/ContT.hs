{-# LANGUAGE ScopedTypeVariables #-}

module Graphics.WebGPU.Dawn.ContT
  ( -- * Context Management
    createContext
  , createContextWithFeatures
  , WGPUFeatureName(..)
    -- * Tensor Management
  , createTensor
  , createTensorWithData
  , createTensorWithDataPacked
  , toGPU
  , fromGPU
    -- * Kernel Management
  , createKernelCode
  , createKernel
  , dispatchKernel
  , dispatchKernelAsync
  , waitAll
  , beginBatch
  , endBatch
  , setWorkgroupSize
  , setEntryPoint
    -- * Nested ContT Helper
  , runContT'
    -- * Re-exports
  , module Graphics.WebGPU.Dawn.Types
  , TensorData
  , WorkgroupSize(..)
  , defaultWorkgroupSize
  , ContT(..)
  , evalContT
  , liftIO
  , lift
  ) where

import Control.Monad.Trans.Cont (ContT(..), evalContT)
import Control.Monad.Trans.Class (lift)
import Control.Monad.IO.Class (liftIO)
import Data.Vector.Storable (Vector, Storable)
import Graphics.WebGPU.Dawn.Types
import Graphics.WebGPU.Dawn.Context (WGPUFeatureName(..))
import qualified Graphics.WebGPU.Dawn.Tensor as T
import Graphics.WebGPU.Dawn.Tensor (TensorData)
import qualified Graphics.WebGPU.Dawn.Kernel as K
import Graphics.WebGPU.Dawn.Kernel (WorkgroupSize(..), defaultWorkgroupSize)
import qualified Graphics.WebGPU.Dawn.IO as IO

-- | Create a context with automatic cleanup using ContT
createContext :: ContT r IO Context
createContext = ContT IO.withContext

-- | Create a context with features and automatic cleanup using ContT
createContextWithFeatures :: [String] -> [WGPUFeatureName] -> ContT r IO Context
createContextWithFeatures toggles features = ContT $ IO.withContextFeatures toggles features

-- | Create a tensor with automatic cleanup using ContT
createTensor :: Context -> Shape -> NumType -> ContT r IO (Tensor dtype)
createTensor ctx shape dtype = ContT $ IO.withTensor ctx shape dtype

-- | Create a tensor with data and automatic cleanup using ContT
createTensorWithData :: forall a r dtype. TensorData a => Context -> Shape -> Vector a -> ContT r IO (Tensor dtype)
createTensorWithData ctx shape vec = ContT $ \k -> IO.withTensorWithData ctx shape vec k

-- | Create a tensor with pre-packed data and explicit NumType with automatic cleanup using ContT
createTensorWithDataPacked :: forall a r dtype. Storable a => Context -> Shape -> NumType -> Vector a -> ContT r IO (Tensor dtype)
createTensorWithDataPacked ctx shape dtype vec = ContT $ \k -> do
  tensor <- T.createTensorWithDataPacked ctx shape dtype vec
  result <- k tensor
  T.destroyTensor tensor
  return result

-- | Create kernel code with automatic cleanup using ContT
createKernelCode :: String -> ContT r IO KernelCode
createKernelCode wgslSource = ContT $ IO.withKernelCode wgslSource

-- | Compile a kernel with automatic cleanup using ContT
createKernel :: Context -> KernelCode -> [Tensor dtype] -> WorkgroupSize -> ContT r IO Kernel
createKernel ctx code tensors wgSize = ContT $ IO.withKernel ctx code tensors wgSize

-- | Transfer data from CPU to GPU (re-exported from Tensor module)
toGPU :: forall a dtype. TensorData a => Context -> Tensor dtype -> Vector a -> IO ()
toGPU = T.toGPU

-- | Transfer data from GPU to CPU (re-exported from Tensor module)
fromGPU :: forall a dtype. TensorData a => Context -> Tensor dtype -> Int -> IO (Vector a)
fromGPU = T.fromGPU

-- | Dispatch (execute) a compiled kernel on the GPU synchronously (re-exported from Kernel module)
-- This waits for the kernel to complete before returning
dispatchKernel :: Context -> Kernel -> IO ()
dispatchKernel = K.dispatchKernel

-- | Dispatch a kernel asynchronously (non-blocking) (re-exported from Kernel module)
-- The kernel executes in the background. Call 'waitAll' to synchronize.
-- This allows GPU pipelining for dramatically better performance!
dispatchKernelAsync :: Context -> Kernel -> IO ()
dispatchKernelAsync = K.dispatchKernelAsync

-- | Wait for all pending async kernel dispatches to complete (re-exported from Kernel module)
-- Call this before reading results from GPU tensors
waitAll :: Context -> IO ()
waitAll = K.waitAll

-- | Begin batching GPU commands (re-exported from Kernel module)
-- All subsequent kernel dispatches will be accumulated and submitted together
beginBatch :: Context -> IO ()
beginBatch = K.beginBatch

-- | End batching and submit all accumulated GPU commands (re-exported from Kernel module)
-- This ensures proper synchronization barriers between dependent operations
endBatch :: Context -> IO ()
endBatch = K.endBatch

-- | Set the workgroup size for kernel code (re-exported from Kernel module)
setWorkgroupSize :: KernelCode -> WorkgroupSize -> IO ()
setWorkgroupSize = K.setWorkgroupSize

-- | Set the entry point function name for the kernel (re-exported from Kernel module)
setEntryPoint :: KernelCode -> String -> IO ()
setEntryPoint = K.setEntryPoint

-- | Helper for nested ContT scopes - evaluates an inner ContT computation within an outer ContT
-- This allows partial resource release: inner scope resources are released when the inner
-- computation completes, while outer scope resources remain alive.
--
-- Example:
-- @
-- evalContT $ do
--   ctx <- createContext                    -- Outer scope
--   runContT' $ do
--     tensor <- createTensor ctx ...        -- Inner scope
--     liftIO $ useResource tensor
--   -- tensor is released here, ctx still alive
--   liftIO $ continueWithContext ctx
-- @
runContT' :: ContT a IO a -> ContT r IO a
runContT' inner = liftIO $ evalContT inner
