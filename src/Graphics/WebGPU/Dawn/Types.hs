{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}

module Graphics.WebGPU.Dawn.Types
  ( -- * Opaque Types
    Context(..)
  , Tensor(..)
  , Kernel(..)
  , KernelCode(..)
    -- * Data Types
  , Shape(..)
  , NumType(..)
  , GPUException(..)
  , Half(..)
    -- * Helper Functions
  , numTypeSize
  , shapeSize
  , shapeDims
  ) where

import Control.Exception (Exception)
import Foreign.Storable
import Foreign.Ptr
import Data.Word
import qualified Graphics.WebGPU.Dawn.Internal as I

-- Re-export NumType from Internal to avoid duplication
import Graphics.WebGPU.Dawn.Internal (NumType(..))

-- | Managed GPU context
newtype Context = Context I.Context
  deriving (Eq)

-- | Managed GPU tensor with phantom type parameter for dtype tracking
-- The type parameter 'dtype' is a phantom type that tracks the data type at compile time
newtype Tensor (dtype :: NumType) = Tensor I.Tensor
  deriving (Eq)

-- | Half-precision floating point (FP16) represented as Word16
newtype Half = Half Word16
  deriving (Eq, Show)

instance Storable Half where
  sizeOf _ = 2
  alignment _ = 2
  peek ptr = Half <$> peek (castPtr ptr :: Ptr Word16)
  poke ptr (Half w) = poke (castPtr ptr :: Ptr Word16) w

-- | Managed GPU kernel
newtype Kernel = Kernel I.Kernel
  deriving (Eq)

-- | Managed kernel code (WGSL shader)
newtype KernelCode = KernelCode I.KernelCode
  deriving (Eq)

-- | Tensor shape (dimensions)
data Shape = Shape [Int]
  deriving (Eq, Show)

-- | Exception type for GPU operations
data GPUException
  = GPUError Int String       -- ^ Error code and message
  | InvalidShape String       -- ^ Invalid shape specification
  | InvalidDataSize String    -- ^ Data size mismatch
  | CompilationError String   -- ^ Shader compilation error
  deriving (Show, Eq)

instance Exception GPUException

-- | Get size in bytes of a numeric type
-- Note: For packed types (I4, U4, I8, U8, I16, U16), this returns the
-- size per element, not the packed size. Use shapeSize to get total elements.
numTypeSize :: NumType -> Int
numTypeSize F16 = 2
numTypeSize F32 = 4
numTypeSize F64 = 8
numTypeSize I4  = 1  -- Packed: 8 per u32, but conceptually 0.5 bytes each
numTypeSize I8  = 1
numTypeSize I16 = 2
numTypeSize I32 = 4
numTypeSize I64 = 8
numTypeSize U4  = 1  -- Packed: 8 per u32, but conceptually 0.5 bytes each
numTypeSize U8  = 1
numTypeSize U16 = 2
numTypeSize U32 = 4
numTypeSize U64 = 8
numTypeSize Unknown = 0

-- | Calculate total number of elements in a shape
shapeSize :: Shape -> Int
shapeSize (Shape dims) = product dims

-- | Get dimensions of a shape
shapeDims :: Shape -> [Int]
shapeDims (Shape dims) = dims
