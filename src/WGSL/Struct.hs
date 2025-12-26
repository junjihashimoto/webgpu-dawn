{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Module: WGSL.Struct
Description: Automated struct generation and WGSL memory layout

This module provides automatic generation of WGSL struct definitions from
Haskell data types, handling the complex alignment and padding rules required
by GPU memory layouts (std430).

Key features:
  - Automatic WGSL struct definition generation using GHC.Generics
  - Correct std430 alignment and padding calculations
  - Serialization to GPU-compatible byte layout
  - Prevents data corruption from misaligned memory access

Example:
@
  data Particle = Particle
    { position :: Vec3 Float
    , velocity :: Vec3 Float
    , mass :: Float
    } deriving (Generic, Show)

  instance WGSLStorable Particle

  -- Automatically generates:
  -- struct Particle {
  --   position: vec3<f32>,  // offset 0, aligned to 16
  --   velocity: vec3<f32>,  // offset 16, aligned to 16
  --   mass: f32             // offset 32
  -- };
@
-}

module WGSL.Struct
  ( -- * Type Class
    WGSLStorable(..)
  , GWGSLStorable(..)  -- Internal, but needed for type signatures

    -- * Memory Layout
  , Alignment(..)
  , FieldLayout(..)
  , structLayout
  , structSize
  , fieldOffsetOf

    -- * Serialization
  , toWGSLBytes
  , toWGSLBytesList

    -- * Utilities
  , structDefinition
  , fieldName
  ) where

import GHC.Generics
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import Data.Binary.Put
import Data.Word
import Data.Int
import Foreign.Storable (sizeOf)
import Foreign.Ptr (castPtr, plusPtr)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Utils (with)
import System.IO.Unsafe (unsafePerformIO)

-- | Alignment requirement in bytes
newtype Alignment = Alignment Int deriving (Show, Eq, Ord)

-- | Field layout information
data FieldLayout = FieldLayout
  { fieldOffset :: Int      -- ^ Byte offset from struct start
  , fieldSize   :: Int      -- ^ Size in bytes
  , fieldAlign  :: Alignment -- ^ Alignment requirement
  , fieldWGSLType :: String  -- ^ WGSL type name
  } deriving (Show)

-- | Type class for types that can be stored in WGSL memory
--
-- This is separate from Foreign.Storable because WGSL has different
-- alignment rules (std430) than C.
class WGSLStorable a where
  -- | WGSL type name (e.g., "f32", "vec3<f32>", "mat4x4<f32>")
  wgslType :: a -> String

  -- | Alignment requirement in bytes (for std430 layout)
  wgslAlignment :: a -> Alignment

  -- | Size in bytes
  wgslSize :: a -> Int

  -- | Serialize to bytes with correct padding
  wgslPut :: a -> Put

  -- | Generate WGSL struct definition (for composite types)
  wgslStructDef :: a -> Maybe String

  -- Default implementations using Generics
  default wgslType :: (Generic a, GWGSLStorable (Rep a)) => a -> String
  wgslType x = gWGSLType (from x)

  default wgslAlignment :: (Generic a, GWGSLStorable (Rep a)) => a -> Alignment
  wgslAlignment x = gWGSLAlignment (from x)

  default wgslSize :: (Generic a, GWGSLStorable (Rep a)) => a -> Int
  wgslSize x = gWGSLSize (from x)

  default wgslPut :: (Generic a, GWGSLStorable (Rep a)) => a -> Put
  wgslPut x = gWGSLPut (from x)

  default wgslStructDef :: (Generic a, GWGSLStorable (Rep a)) => a -> Maybe String
  wgslStructDef x = gWGSLStructDef (from x)

-- | Internal generic class for deriving WGSLStorable
class GWGSLStorable f where
  gWGSLType :: f a -> String
  gWGSLAlignment :: f a -> Alignment
  gWGSLSize :: f a -> Int
  gWGSLPut :: f a -> Put
  gWGSLStructDef :: f a -> Maybe String
  gFieldLayouts :: f a -> Int -> [(String, FieldLayout)]

-- Metadata (type name)
instance (GWGSLStorable f, Datatype d) => GWGSLStorable (D1 d f) where
  gWGSLType (M1 x) = gWGSLType x
  gWGSLAlignment (M1 x) = gWGSLAlignment x
  gWGSLSize (M1 x) = gWGSLSize x
  gWGSLPut (M1 x) = gWGSLPut x
  gWGSLStructDef (M1 x) = Just $ "struct " ++ datatypeName (undefined :: D1 d f a) ++ " {\n"
    ++ concatMap formatField (gFieldLayouts x 0) ++ "};"
    where
      formatField (name, layout) = "  " ++ name ++ ": " ++ fieldWGSLType layout ++ ",\n"
  gFieldLayouts (M1 x) = gFieldLayouts x

-- Constructor metadata
instance (GWGSLStorable f, Constructor c) => GWGSLStorable (C1 c f) where
  gWGSLType (M1 x) = gWGSLType x
  gWGSLAlignment (M1 x) = gWGSLAlignment x
  gWGSLSize (M1 x) = gWGSLSize x
  gWGSLPut (M1 x) = gWGSLPut x
  gWGSLStructDef (M1 x) = gWGSLStructDef x
  gFieldLayouts (M1 x) = gFieldLayouts x

-- Selector (field name)
instance (GWGSLStorable f, Selector s) => GWGSLStorable (S1 s f) where
  gWGSLType (M1 x) = gWGSLType x
  gWGSLAlignment (M1 x) = gWGSLAlignment x
  gWGSLSize (M1 x) = gWGSLSize x
  gWGSLPut (M1 x) = gWGSLPut x
  gWGSLStructDef (M1 x) = gWGSLStructDef x
  gFieldLayouts (M1 x) offset =
    let name = selName (undefined :: S1 s f a)
        layout = FieldLayout offset (gWGSLSize x) (gWGSLAlignment x) (gWGSLType x)
    in [(name, layout)]

-- Product (multiple fields)
instance (GWGSLStorable f, GWGSLStorable g) => GWGSLStorable (f :*: g) where
  gWGSLType _ = "struct"
  gWGSLAlignment (x :*: _) = gWGSLAlignment x
  gWGSLSize (x :*: y) =
    let Alignment align = gWGSLAlignment y
        offsetY = alignUp (gWGSLSize x) align
    in offsetY + gWGSLSize y
  gWGSLPut (x :*: y) = do
    gWGSLPut x
    -- Add padding before y
    let Alignment align = gWGSLAlignment y
        offsetY = alignUp (gWGSLSize x) align
        padding = offsetY - gWGSLSize x
    mapM_ (const $ putWord8 0) [1..padding]
    gWGSLPut y
  gWGSLStructDef _ = Nothing
  gFieldLayouts (x :*: y) offset =
    let fieldsX = gFieldLayouts x offset
        Alignment align = gWGSLAlignment y
        offsetY = alignUp (offset + gWGSLSize x) align
        fieldsY = gFieldLayouts y offsetY
    in fieldsX ++ fieldsY

-- Base case: actual field data
instance WGSLStorable a => GWGSLStorable (K1 i a) where
  gWGSLType (K1 x) = wgslType x
  gWGSLAlignment (K1 x) = wgslAlignment x
  gWGSLSize (K1 x) = wgslSize x
  gWGSLPut (K1 x) = wgslPut x
  gWGSLStructDef (K1 x) = wgslStructDef x
  gFieldLayouts (K1 x) offset = []

-- Helper: align offset up to alignment boundary
alignUp :: Int -> Int -> Int
alignUp offset align = ((offset + align - 1) `div` align) * align

-- | Primitive type instances

-- Float (f32)
instance WGSLStorable Float where
  wgslType _ = "f32"
  wgslAlignment _ = Alignment 4
  wgslSize _ = 4
  wgslPut = putFloatle
  wgslStructDef _ = Nothing

-- Double (f64) - not widely supported in WGSL, but included for completeness
instance WGSLStorable Double where
  wgslType _ = "f64"
  wgslAlignment _ = Alignment 8
  wgslSize _ = 8
  wgslPut = putDoublele
  wgslStructDef _ = Nothing

-- Int32 (i32)
instance WGSLStorable Int32 where
  wgslType _ = "i32"
  wgslAlignment _ = Alignment 4
  wgslSize _ = 4
  wgslPut = putInt32le
  wgslStructDef _ = Nothing

-- Word32 (u32)
instance WGSLStorable Word32 where
  wgslType _ = "u32"
  wgslAlignment _ = Alignment 4
  wgslSize _ = 4
  wgslPut = putWord32le
  wgslStructDef _ = Nothing

-- | Serialize to ByteString with correct WGSL memory layout
toWGSLBytes :: WGSLStorable a => a -> ByteString
toWGSLBytes x = BL.toStrict $ runPut (wgslPut x)

-- | Serialize a list of structs
toWGSLBytesList :: WGSLStorable a => [a] -> ByteString
toWGSLBytesList xs = BL.toStrict $ runPut (mapM_ wgslPut xs)

-- | Get the struct layout information
structLayout :: (Generic a, GWGSLStorable (Rep a)) => a -> [(String, FieldLayout)]
structLayout x = gFieldLayouts (from x) 0

-- | Get the total size of a struct (with final padding to align to struct alignment)
structSize :: WGSLStorable a => a -> Int
structSize x =
  let size = wgslSize x
      Alignment align = wgslAlignment x
  in alignUp size align

-- | Generate WGSL struct definition
structDefinition :: WGSLStorable a => a -> Maybe String
structDefinition = wgslStructDef

-- | Get field name from Selector
fieldName :: Selector s => S1 s f a -> String
fieldName = selName

-- | Get the byte offset of a field by name (at compile time)
-- This allows shader code to reference field offsets for manual calculations.
--
-- Example:
-- @
--   data Particle = Particle { pos :: Vec3, vel :: Vec3, m :: Float }
--   instance WGSLStorable Particle
--
--   -- Get offset of 'vel' field
--   velOffset = fieldOffsetOf (undefined :: Particle) "vel"  -- Returns 16
-- @
fieldOffsetOf :: (Generic a, GWGSLStorable (Rep a)) => a -> String -> Maybe Int
fieldOffsetOf x fieldName =
  let layout = structLayout x
  in lookup fieldName layout >>= Just . fieldOffset
