module Graphics.WebGPU.Dawn.Kernel
  ( -- * Kernel Code
    KernelCode
  , createKernelCode
  , setWorkgroupSize
  , setEntryPoint
  , destroyKernelCode
    -- * Kernel Compilation
  , Kernel
  , compileKernel
  , compileKernelHeterogeneous
  , destroyKernel
    -- * Kernel Execution
  , dispatchKernel
  , dispatchKernelAsync
  , waitAll
    -- * Command Batching
  , beginBatch
  , endBatch
    -- * Workgroup Configuration
  , WorkgroupSize(..)
  , defaultWorkgroupSize
  ) where

import Control.Exception (throwIO)
import Foreign
import Foreign.C.Types
import Foreign.C.String
import qualified Graphics.WebGPU.Dawn.Internal as I
import Graphics.WebGPU.Dawn.Types
import Graphics.WebGPU.Dawn.Context (checkError)

-- | Workgroup size for kernel dispatch
data WorkgroupSize = WorkgroupSize
  { workgroupX :: Int
  , workgroupY :: Int
  , workgroupZ :: Int
  } deriving (Eq, Show)

-- | Default workgroup size (256, 1, 1)
defaultWorkgroupSize :: WorkgroupSize
defaultWorkgroupSize = WorkgroupSize 256 1 1

-- | Create kernel code from WGSL shader source
createKernelCode :: String -> IO KernelCode
createKernelCode wgslSource = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)

  code <- withCString wgslSource $ \cstr ->
    I.c_createKernelCode cstr errPtr

  checkError errPtr

  if code == nullPtr
    then throwIO $ CompilationError "Failed to create kernel code"
    else return $ KernelCode code

-- | Set the workgroup size for kernel code
setWorkgroupSize :: KernelCode -> WorkgroupSize -> IO ()
setWorkgroupSize (KernelCode code) (WorkgroupSize x y z) =
  I.c_setKernelWorkgroupSize code (fromIntegral x) (fromIntegral y) (fromIntegral z)

-- | Set the entry point function name for the kernel
setEntryPoint :: KernelCode -> String -> IO ()
setEntryPoint (KernelCode code) entryPoint =
  withCString entryPoint $ \cstr ->
    I.c_setKernelEntryPoint code cstr

-- | Destroy kernel code and free resources
destroyKernelCode :: KernelCode -> IO ()
destroyKernelCode (KernelCode code) = I.c_destroyKernelCode code

-- | Compile a kernel from kernel code with bound tensors
-- Note: Tensors can have different dtype parameters, so we use existential types here
compileKernel
  :: Context
  -> KernelCode
  -> [Tensor dtype]           -- ^ Input/output tensors to bind (all same dtype for now)
  -> WorkgroupSize      -- ^ Number of workgroups to dispatch
  -> IO Kernel
compileKernel (Context ctx) (KernelCode code) tensors wgSize = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)

  let tensorPtrs = map (\(Tensor t) -> t) tensors
      numTensors = length tensors
      WorkgroupSize wx wy wz = wgSize

  kernel <- withArray tensorPtrs $ \tensorArr ->
    I.c_createKernel
      ctx
      code
      tensorArr
      (fromIntegral numTensors)
      (fromIntegral wx)
      (fromIntegral wy)
      (fromIntegral wz)
      nullPtr      -- cache_key: nullptr enables auto-generation in C++
      errPtr

  checkError errPtr

  if kernel == nullPtr
    then throwIO $ CompilationError "Failed to compile kernel"
    else return $ Kernel kernel

-- | Compile a kernel with heterogeneous tensors (different dtypes)
-- This allows mixing F32, I32, U32, etc. tensors in the same kernel
-- This is the safe version that should be used with executeShaderNamed
compileKernelHeterogeneous
  :: Context
  -> KernelCode
  -> [AnyTensor]              -- ^ Input/output tensors with potentially different dtypes
  -> WorkgroupSize
  -> IO Kernel
compileKernelHeterogeneous (Context ctx) (KernelCode code) anyTensors wgSize = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)

  -- Extract internal tensor pointers from existential wrappers
  let tensorPtrs = map (\(AnyTensor (Tensor t)) -> t) anyTensors
      numTensors = length anyTensors
      WorkgroupSize wx wy wz = wgSize

  kernel <- withArray tensorPtrs $ \tensorArr ->
    I.c_createKernel
      ctx
      code
      tensorArr
      (fromIntegral numTensors)
      (fromIntegral wx)
      (fromIntegral wy)
      (fromIntegral wz)
      nullPtr      -- cache_key: nullptr enables auto-generation in C++
      errPtr

  checkError errPtr

  if kernel == nullPtr
    then throwIO $ CompilationError "Failed to compile kernel"
    else return $ Kernel kernel

-- | Destroy a compiled kernel
destroyKernel :: Kernel -> IO ()
destroyKernel (Kernel kernel) = I.c_destroyKernel kernel

-- | Dispatch (execute) a compiled kernel on the GPU (synchronous)
-- This waits for the kernel to complete before returning
dispatchKernel :: Context -> Kernel -> IO ()
dispatchKernel (Context ctx) (Kernel kernel) = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  I.c_dispatchKernel ctx kernel errPtr
  checkError errPtr

-- | Dispatch a kernel asynchronously (non-blocking)
-- The kernel executes in the background. Call 'waitAll' to synchronize.
-- This allows GPU pipelining for much better performance!
dispatchKernelAsync :: Context -> Kernel -> IO ()
dispatchKernelAsync (Context ctx) (Kernel kernel) = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  I.c_dispatchKernelAsync ctx kernel errPtr
  checkError errPtr

-- | Wait for all pending async kernel dispatches to complete
-- Call this before reading results from GPU tensors
waitAll :: Context -> IO ()
waitAll (Context ctx) = I.c_waitAll ctx

-- | Begin batching GPU commands - all subsequent kernel dispatches will be accumulated
-- into a single command encoder and submitted together when endBatch is called.
-- This ensures proper synchronization barriers between dependent kernels.
beginBatch :: Context -> IO ()
beginBatch (Context ctx) = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  I.c_beginBatch ctx errPtr
  checkError errPtr

-- | End batching and submit all accumulated GPU commands in a single submission.
-- WebGPU automatically inserts memory barriers between compute passes to ensure
-- proper synchronization of dependent operations.
endBatch :: Context -> IO ()
endBatch (Context ctx) = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  I.c_endBatch ctx errPtr
  checkError errPtr
