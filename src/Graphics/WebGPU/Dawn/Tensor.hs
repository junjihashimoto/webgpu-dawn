{-# LANGUAGE ScopedTypeVariables #-}

module Graphics.WebGPU.Dawn.Tensor
  ( -- * Tensor Creation
    createTensor
  , createTensorWithData
  , destroyTensor
    -- * Data Transfer
  , toGPU
  , fromGPU
    -- * Storable Type Class Helpers
  , TensorData(..)
  ) where

import Control.Exception (throwIO)
import Foreign
import Foreign.C.Types
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

-- | Create an empty tensor with the given shape and data type
createTensor :: Context -> Shape -> NumType -> IO Tensor
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
createTensorWithData :: forall a. TensorData a => Context -> Shape -> Vector a -> IO Tensor
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

      -- Clean up shape
      I.c_destroyShape shapePtr

      if tensor == nullPtr
        then throwIO $ GPUError 1 "Failed to create tensor with data"
        else return $ Tensor tensor

-- | Destroy a tensor and free its GPU memory
destroyTensor :: Tensor -> IO ()
destroyTensor (Tensor tensor) = I.c_destroyTensor tensor

-- | Transfer data from CPU to GPU
toGPU :: forall a. TensorData a => Context -> Tensor -> Vector a -> IO ()
toGPU (Context ctx) (Tensor tensor) vec = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)

  let size = V.length vec * sizeOf (undefined :: a)

  V.unsafeWith vec $ \dataPtr ->
    I.c_toGPU ctx (castPtr dataPtr) tensor (fromIntegral size) errPtr

  checkError errPtr

-- | Transfer data from GPU to CPU
fromGPU :: forall a. TensorData a => Context -> Tensor -> Int -> IO (Vector a)
fromGPU (Context ctx) (Tensor tensor) numElements = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)

  let size = numElements * sizeOf (undefined :: a)

  -- Allocate mutable vector and read data into it
  allocaBytes size $ \dataPtr -> do
    I.c_toCPU ctx tensor dataPtr (fromIntegral size) errPtr
    checkError errPtr
    -- Generate vector from the data we just read
    V.generateM numElements $ \i -> peekElemOff (castPtr dataPtr :: Ptr a) i
