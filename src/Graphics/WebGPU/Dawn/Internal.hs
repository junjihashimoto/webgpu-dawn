{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE EmptyDataDecls #-}
{-# LANGUAGE CApiFFI #-}

module Graphics.WebGPU.Dawn.Internal where

import Foreign
import Foreign.C.Types
import Foreign.C.String
import Data.Word (Word32, Word64)

-- Opaque types corresponding to C wrapper
data Context_
data Tensor_
data Kernel_
data Shape_
data KernelCode_
data QuerySet_
data CommandEncoder_
data Buffer_

-- Type aliases for opaque pointers
type Context = Ptr Context_
type Tensor = Ptr Tensor_
type Kernel = Ptr Kernel_
type Shape = Ptr Shape_
type KernelCode = Ptr KernelCode_
type QuerySet = Ptr QuerySet_
type CommandEncoder = Ptr CommandEncoder_
type Buffer = Ptr Buffer_

-- Error handling structure
data GPUError = GPUError
  { errorCode :: CInt
  , errorMessage :: CString
  }

instance Storable GPUError where
  sizeOf _ = sizeOf (undefined :: CInt) + sizeOf (undefined :: CString)
  alignment _ = alignment (undefined :: CInt)
  peek ptr = do
    code <- peekByteOff ptr 0
    msg <- peekByteOff ptr (sizeOf (undefined :: CInt))
    return $ GPUError code msg
  poke ptr (GPUError code msg) = do
    pokeByteOff ptr 0 code
    pokeByteOff ptr (sizeOf (undefined :: CInt)) msg

-- Numeric type enum (matches Types.hs)
data NumType
  = F16
  | F32
  | F64
  | I4   -- 4-bit signed (packed: 8 nibbles per u32)
  | I8
  | I16
  | I32
  | I64
  | U4   -- 4-bit unsigned (packed: 8 nibbles per u32)
  | U8
  | U16
  | U32
  | U64
  | Unknown
  deriving (Eq, Show, Enum, Bounded)

-- Convert NumType to CInt
numTypeToCInt :: NumType -> CInt
numTypeToCInt = fromIntegral . fromEnum

-- Context management
foreign import ccall unsafe "gpu_create_context"
  c_createContext :: Ptr GPUError -> IO Context

foreign import ccall unsafe "gpu_create_context_with_features"
  c_createContextWithFeatures :: Ptr CString -> CSize -> Ptr Word32 -> CSize -> Ptr GPUError -> IO Context

foreign import ccall unsafe "gpu_destroy_context"
  c_destroyContext :: Context -> IO ()

-- Shape management
foreign import ccall unsafe "gpu_create_shape"
  c_createShape :: Ptr CSize -> CSize -> Ptr GPUError -> IO Shape

foreign import ccall unsafe "gpu_destroy_shape"
  c_destroyShape :: Shape -> IO ()

foreign import ccall unsafe "gpu_shape_size"
  c_shapeSize :: Shape -> IO CSize

foreign import ccall unsafe "gpu_shape_rank"
  c_shapeRank :: Shape -> IO CSize

foreign import ccall unsafe "gpu_shape_dim"
  c_shapeDim :: Shape -> CSize -> IO CSize

-- Tensor management
foreign import ccall unsafe "gpu_create_tensor"
  c_createTensor :: Context -> Shape -> CInt -> Ptr GPUError -> IO Tensor

foreign import ccall unsafe "gpu_create_tensor_with_data"
  c_createTensorWithData :: Context -> Shape -> CInt -> Ptr () -> CSize -> Ptr GPUError -> IO Tensor

foreign import ccall unsafe "gpu_destroy_tensor"
  c_destroyTensor :: Tensor -> IO ()

foreign import ccall unsafe "gpu_tensor_size_bytes"
  c_tensorSizeBytes :: Tensor -> IO CSize

-- Data transfer
foreign import ccall unsafe "gpu_to_cpu"
  c_toCPU :: Context -> Tensor -> Ptr () -> CSize -> Ptr GPUError -> IO ()

foreign import ccall unsafe "gpu_to_gpu"
  c_toGPU :: Context -> Ptr () -> Tensor -> CSize -> Ptr GPUError -> IO ()

-- Kernel code management
foreign import ccall unsafe "gpu_create_kernel_code"
  c_createKernelCode :: CString -> Ptr GPUError -> IO KernelCode

foreign import ccall unsafe "gpu_set_kernel_workgroup_size"
  c_setKernelWorkgroupSize :: KernelCode -> CSize -> CSize -> CSize -> IO ()

foreign import ccall unsafe "gpu_set_kernel_entry_point"
  c_setKernelEntryPoint :: KernelCode -> CString -> IO ()

foreign import ccall unsafe "gpu_destroy_kernel_code"
  c_destroyKernelCode :: KernelCode -> IO ()

-- Kernel compilation and execution
foreign import ccall unsafe "gpu_create_kernel"
  c_createKernel :: Context -> KernelCode -> Ptr Tensor -> CSize -> CSize -> CSize -> CSize -> Ptr CChar -> Ptr GPUError -> IO Kernel

foreign import ccall unsafe "gpu_destroy_kernel"
  c_destroyKernel :: Kernel -> IO ()

foreign import ccall unsafe "gpu_dispatch_kernel"
  c_dispatchKernel :: Context -> Kernel -> Ptr GPUError -> IO ()

foreign import ccall unsafe "gpu_dispatch_kernel_async"
  c_dispatchKernelAsync :: Context -> Kernel -> Ptr GPUError -> IO ()

foreign import ccall unsafe "gpu_wait_all"
  c_waitAll :: Context -> IO ()

foreign import ccall unsafe "gpu_begin_batch"
  c_beginBatch :: Context -> Ptr GPUError -> IO ()

foreign import ccall unsafe "gpu_end_batch"
  c_endBatch :: Context -> Ptr GPUError -> IO ()

-- Utility functions
foreign import ccall unsafe "gpu_size_of_type"
  c_sizeOfType :: CInt -> IO CSize

foreign import ccall unsafe "gpu_type_name"
  c_typeName :: CInt -> IO CString

foreign import ccall unsafe "gpu_has_error"
  c_hasError :: Ptr GPUError -> IO CInt

foreign import ccall unsafe "gpu_get_last_error_message"
  c_getLastErrorMessage :: Ptr GPUError -> IO CString

foreign import ccall unsafe "gpu_clear_error"
  c_clearError :: Ptr GPUError -> IO ()

-- Timestamp query support
foreign import ccall unsafe "gpu_create_query_set"
  c_createQuerySet :: Context -> CInt -> Word32 -> Ptr GPUError -> IO QuerySet

foreign import ccall unsafe "gpu_destroy_query_set"
  c_destroyQuerySet :: QuerySet -> IO ()

foreign import ccall unsafe "gpu_get_command_encoder"
  c_getCommandEncoder :: Context -> Ptr GPUError -> IO CommandEncoder

foreign import ccall unsafe "gpu_release_command_encoder"
  c_releaseCommandEncoder :: CommandEncoder -> IO ()

foreign import ccall unsafe "gpu_write_timestamp"
  c_writeTimestamp :: CommandEncoder -> QuerySet -> Word32 -> IO ()

foreign import ccall unsafe "gpu_resolve_query_set"
  c_resolveQuerySet :: CommandEncoder -> QuerySet -> Word32 -> Word32 -> Buffer -> Word64 -> IO ()

foreign import ccall unsafe "gpu_create_query_buffer"
  c_createQueryBuffer :: Context -> CSize -> Ptr GPUError -> IO Buffer

foreign import ccall unsafe "gpu_release_buffer"
  c_releaseBuffer :: Buffer -> IO ()

foreign import ccall unsafe "gpu_read_query_buffer"
  c_readQueryBuffer :: Context -> Buffer -> Ptr Word64 -> CSize -> Ptr GPUError -> IO ()

-- Debug Ring Buffer (for GPU printf)
data DebugBuffer_
type DebugBuffer = Ptr DebugBuffer_

foreign import ccall unsafe "gpu_create_debug_buffer"
  c_createDebugBuffer :: Context -> CSize -> Ptr GPUError -> IO DebugBuffer

foreign import ccall unsafe "gpu_destroy_debug_buffer"
  c_destroyDebugBuffer :: DebugBuffer -> IO ()

foreign import ccall unsafe "gpu_read_debug_buffer"
  c_readDebugBuffer :: Context -> DebugBuffer -> Ptr Word32 -> CSize -> Ptr GPUError -> IO Word32

foreign import ccall unsafe "gpu_clear_debug_buffer"
  c_clearDebugBuffer :: Context -> DebugBuffer -> IO ()
