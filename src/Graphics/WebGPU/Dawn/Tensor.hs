{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE KindSignatures #-}

module Graphics.WebGPU.Dawn.Tensor
  ( -- * Tensor Creation
    createTensor
  , createTensorWithData
  , createTensorWithDataPacked
  , destroyTensor
    -- * Typed Tensor Creation (with phantom types)
  , createTensorTyped
  , createTensorWithDataTyped
    -- * Data Transfer
  , toGPU
  , fromGPU
    -- * Type Class Helpers
  , TensorData(..)
  , HasNumType(..)
  ) where

import Control.Exception (throwIO)
import Foreign
import Foreign.C.Types
import Data.Proxy
import Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V
import qualified Graphics.WebGPU.Dawn.Internal as I
import Graphics.WebGPU.Dawn.Types
import Graphics.WebGPU.Dawn.Context (checkError)

-- | Type class for data that can be stored in tensors
class Storable a => TensorData a where
  tensorNumType :: proxy a -> NumType

instance TensorData Float where
  tensorNumType _ = F32

instance TensorData Double where
  tensorNumType _ = F64

instance TensorData Int32 where
  tensorNumType _ = I32

instance TensorData Word32 where
  tensorNumType _ = U32

instance TensorData Word16 where
  tensorNumType _ = F16

instance TensorData Half where
  tensorNumType _ = F16

-- | Type class mapping Haskell types to NumType phantom parameters
-- This allows us to write: createTensorTyped :: Vector Float -> IO (Tensor 'F32)
class (Storable a, TensorData a) => HasNumType (dtype :: NumType) a | dtype -> a, a -> dtype where
  getDType :: proxy dtype -> NumType

instance HasNumType 'F32 Float where
  getDType _ = F32

instance HasNumType 'F16 Half where
  getDType _ = F16

instance HasNumType 'F64 Double where
  getDType _ = F64

instance HasNumType 'I32 Int32 where
  getDType _ = I32

instance HasNumType 'U32 Word32 where
  getDType _ = U32

-- | Create an empty tensor with the given shape and data type
-- Note: This returns an untyped Tensor. Use createTensorTyped for type-safe version.
createTensor :: Context -> Shape -> NumType -> IO (Tensor dtype)
createTensor (Context ctx) shape@(Shape dims) dtype = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)

  -- Create shape object
  let rank = length dims
  shapePtr <- withArray (map fromIntegral dims) $ \dimPtr ->
    I.c_createShape dimPtr (fromIntegral rank) errPtr
  checkError errPtr

  -- Create tensor
  tensor <- I.c_createTensor ctx shapePtr (I.numTypeToCInt dtype) errPtr
  checkError errPtr

  -- Clean up shape
  I.c_destroyShape shapePtr

  if tensor == nullPtr
    then throwIO $ GPUError 1 "Failed to create tensor"
    else return $ Tensor tensor

-- | Create a tensor initialized with data from a Storable vector
-- Note: This returns an untyped Tensor. Use createTensorWithDataTyped for type-safe version.
createTensorWithData :: forall a dtype. TensorData a => Context -> Shape -> Vector a -> IO (Tensor dtype)
createTensorWithData (Context ctx) shape@(Shape dims) vec = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)

  let dtype = tensorNumType (Nothing :: Maybe a)
      expectedSize = shapeSize shape
      actualSize = V.length vec

  if expectedSize /= actualSize
    then throwIO $ InvalidDataSize $
           "Shape size (" ++ show expectedSize ++ ") != data size (" ++ show actualSize ++ ")"
    else do
      -- Create shape object
      let rank = length dims
      shapePtr <- withArray (map fromIntegral dims) $ \dimPtr ->
        I.c_createShape dimPtr (fromIntegral rank) errPtr
      checkError errPtr

      -- Create tensor with data
      tensor <- V.unsafeWith vec $ \dataPtr ->
        I.c_createTensorWithData
          ctx
          shapePtr
          (I.numTypeToCInt dtype)
          (castPtr dataPtr)
          (fromIntegral $ actualSize * sizeOf (undefined :: a))
          errPtr
      checkError errPtr

      -- CRITICAL FIX: Force GPU to complete async buffer write before Vector can be GC'd
      -- V.unsafeWith only guarantees pointer validity during the callback, but WebGPU's
      -- writeBuffer queues an async operation. Without this sync, Haskell GC can free
      -- the Vector memory before GPU completes the copy, causing use-after-free corruption.
      -- This fixes all 50+ call sites automatically. See: USE_AFTER_FREE_BUG_REPORT.md
      I.c_waitAll ctx

      -- Clean up shape
      I.c_destroyShape shapePtr

      if tensor == nullPtr
        then throwIO $ GPUError 1 "Failed to create tensor with data"
        else return $ Tensor tensor

-- | Create a tensor with pre-packed data and explicit NumType
-- This is used for packed types (I4, U4, I8, U8, I16, U16) where the
-- data is already packed into Word32 arrays but needs a specific NumType.
--
-- For example: Q4 quantization packs 8 nibbles into each Word32, but we
-- need to specify dtype=U4 (not U32) so the buffer size is calculated correctly.
--
-- NOTE: For packed types, the shape represents the logical number of elements
-- (e.g., 16 nibbles for U4), but the data vector contains packed words
-- (e.g., 2 uint32 words). No size validation is performed.
createTensorWithDataPacked :: forall a dtype. Storable a
                           => Context -> Shape -> NumType -> Vector a -> IO (Tensor dtype)
createTensorWithDataPacked (Context ctx) shape@(Shape dims) dtype vec = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)

  let actualSize = V.length vec

  -- No size validation for packed types - shape is logical size, vec is packed size
  do
      -- Create shape object
      let rank = length dims
      shapePtr <- withArray (map fromIntegral dims) $ \dimPtr ->
        I.c_createShape dimPtr (fromIntegral rank) errPtr
      checkError errPtr

      -- Create tensor with data using explicit dtype
      tensor <- V.unsafeWith vec $ \dataPtr ->
        I.c_createTensorWithData
          ctx
          shapePtr
          (I.numTypeToCInt dtype)
          (castPtr dataPtr)
          (fromIntegral $ actualSize * sizeOf (undefined :: a))
          errPtr
      checkError errPtr

      -- CRITICAL FIX: Force GPU to complete async buffer write before Vector can be GC'd
      -- Same use-after-free issue as createTensorWithData. See: USE_AFTER_FREE_BUG_REPORT.md
      I.c_waitAll ctx

      -- Clean up shape
      I.c_destroyShape shapePtr

      if tensor == nullPtr
        then throwIO $ GPUError 1 "Failed to create tensor with data (packed)"
        else return $ Tensor tensor

-- | Create an empty tensor with phantom type tracking (typed version)
-- Usage: createTensorTyped @'F32 ctx (Shape [10, 20])
createTensorTyped :: forall (dtype :: NumType) a. HasNumType dtype a
                  => Context -> Shape -> IO (Tensor dtype)
createTensorTyped (Context ctx) shape@(Shape dims) = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)

  let dtype = getDType (Proxy :: Proxy dtype)
      rank = length dims

  -- Create shape object
  shapePtr <- withArray (map fromIntegral dims) $ \dimPtr ->
    I.c_createShape dimPtr (fromIntegral rank) errPtr
  checkError errPtr

  -- Create tensor
  tensor <- I.c_createTensor ctx shapePtr (I.numTypeToCInt dtype) errPtr
  checkError errPtr

  -- Clean up shape
  I.c_destroyShape shapePtr

  if tensor == nullPtr
    then throwIO $ GPUError 1 "Failed to create typed tensor"
    else return $ Tensor tensor

-- | Create a tensor with data and phantom type tracking (typed version)
-- Usage: createTensorWithDataTyped ctx (Shape [10, 20]) vectorData
-- The type of vectorData (Vector Float or Vector Half) determines the tensor type
createTensorWithDataTyped :: forall (dtype :: NumType) a. HasNumType dtype a
                          => Context -> Shape -> Vector a -> IO (Tensor dtype)
createTensorWithDataTyped (Context ctx) shape@(Shape dims) vec = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)

  let dtype = getDType (Proxy :: Proxy dtype)
      expectedSize = shapeSize shape
      actualSize = V.length vec

  if expectedSize /= actualSize
    then throwIO $ InvalidDataSize $
           "Shape size (" ++ show expectedSize ++ ") != data size (" ++ show actualSize ++ ")"
    else do
      -- Create shape object
      let rank = length dims
      shapePtr <- withArray (map fromIntegral dims) $ \dimPtr ->
        I.c_createShape dimPtr (fromIntegral rank) errPtr
      checkError errPtr

      -- Create tensor with data
      tensor <- V.unsafeWith vec $ \dataPtr ->
        I.c_createTensorWithData
          ctx
          shapePtr
          (I.numTypeToCInt dtype)
          (castPtr dataPtr)
          (fromIntegral $ actualSize * sizeOf (undefined :: a))
          errPtr
      checkError errPtr

      -- Clean up shape
      I.c_destroyShape shapePtr

      if tensor == nullPtr
        then throwIO $ GPUError 1 "Failed to create typed tensor with data"
        else return $ Tensor tensor

-- | Destroy a tensor and free its GPU memory
destroyTensor :: Tensor dtype -> IO ()
destroyTensor (Tensor tensor) = I.c_destroyTensor tensor

-- | Transfer data from CPU to GPU
toGPU :: forall a dtype. TensorData a => Context -> Tensor dtype -> Vector a -> IO ()
toGPU (Context ctx) (Tensor tensor) vec = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)

  let size = V.length vec * sizeOf (undefined :: a)

  V.unsafeWith vec $ \dataPtr ->
    I.c_toGPU ctx (castPtr dataPtr) tensor (fromIntegral size) errPtr

  checkError errPtr

-- | Transfer data from GPU to CPU
fromGPU :: forall a dtype. TensorData a => Context -> Tensor dtype -> Int -> IO (Vector a)
fromGPU (Context ctx) (Tensor tensor) numElements = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)

  let size = numElements * sizeOf (undefined :: a)

  -- Allocate mutable vector and read data into it
  allocaBytes size $ \dataPtr -> do
    I.c_toCPU ctx tensor dataPtr (fromIntegral size) errPtr
    checkError errPtr
    -- Generate vector from the data we just read
    V.generateM numElements $ \i -> peekElemOff (castPtr dataPtr :: Ptr a) i
