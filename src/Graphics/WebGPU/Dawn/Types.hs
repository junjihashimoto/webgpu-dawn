{-# LANGUAGE GeneralizedNewtypeDeriving #-}

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
    -- * Helper Functions
  , numTypeSize
  , shapeSize
  , shapeDims
  ) where

import Control.Exception (Exception)
import qualified Graphics.WebGPU.Dawn.Internal as I

-- Re-export NumType from Internal to avoid duplication
import Graphics.WebGPU.Dawn.Internal (NumType(..))

-- | Managed GPU context
newtype Context = Context I.Context
  deriving (Eq)

-- | Managed GPU tensor
newtype Tensor = Tensor I.Tensor
  deriving (Eq)

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
numTypeSize :: NumType -> Int
numTypeSize F16 = 2
numTypeSize F32 = 4
numTypeSize F64 = 8
numTypeSize I8  = 1
numTypeSize I16 = 2
numTypeSize I32 = 4
numTypeSize I64 = 8
numTypeSize U8  = 1
numTypeSize U16 = 2
numTypeSize U32 = 4
numTypeSize U64 = 8

-- | Calculate total number of elements in a shape
shapeSize :: Shape -> Int
shapeSize (Shape dims) = product dims

-- | Get dimensions of a shape
shapeDims :: Shape -> [Int]
shapeDims (Shape dims) = dims
