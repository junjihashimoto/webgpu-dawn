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
  , destroyKernel
    -- * Kernel Execution
  , dispatchKernel
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
compileKernel
  :: Context
  -> KernelCode
  -> [Tensor]           -- ^ Input/output tensors to bind
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
      errPtr

  checkError errPtr

  if kernel == nullPtr
    then throwIO $ CompilationError "Failed to compile kernel"
    else return $ Kernel kernel

-- | Destroy a compiled kernel
destroyKernel :: Kernel -> IO ()
destroyKernel (Kernel kernel) = I.c_destroyKernel kernel

-- | Dispatch (execute) a compiled kernel on the GPU
dispatchKernel :: Context -> Kernel -> IO ()
dispatchKernel (Context ctx) (Kernel kernel) = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  I.c_dispatchKernel ctx kernel errPtr
  checkError errPtr
