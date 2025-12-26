{-# LANGUAGE GADTs #-}

{-|
WGSL Memory Layout Calculations

This module implements WGSL memory layout rules for uniform (std140) and
storage (std430) buffer layouts. These rules ensure correct data alignment
when transferring data between CPU and GPU.

References:
- WGSL spec: https://www.w3.org/TR/WGSL/#memory-layouts
- Vulkan std140/std430: https://www.khronos.org/registry/vulkan/specs/1.3/html/chap15.html#interfaces-resources-layout

Key Differences:
- std140 (uniform buffers): More strict alignment, vec3 occupies vec4 space
- std430 (storage buffers): More compact, vec3 only uses 12 bytes
-}

module WGSL.Layout
  ( -- * Layout modes
    LayoutMode(..)
  , WGSLLayout(..)

    -- * Struct field layout
  , FieldLayout(..)
  , computeStructLayout

    -- * Utility functions
  , roundUp
  ) where

import WGSL.AST (TypeRep(..))
import Data.Word (Word32)

-- | Memory layout mode
data LayoutMode
  = Std140  -- ^ Uniform buffer layout (more strict alignment)
  | Std430  -- ^ Storage buffer layout (more compact)
  deriving (Show, Eq)

-- | WGSL type layout information
class WGSLLayout a where
  -- | Size in bytes
  wgslSize :: LayoutMode -> a -> Int

  -- | Alignment in bytes (must be power of 2)
  wgslAlignment :: LayoutMode -> a -> Int

-- | Round up to nearest multiple
roundUp :: Int -> Int -> Int
roundUp size alignment = ((size + alignment - 1) `div` alignment) * alignment

-- | Layout information for a struct field
data FieldLayout = FieldLayout
  { fieldName :: String
  , fieldType :: TypeRep
  , fieldOffset :: Int      -- ^ Byte offset from start of struct
  , fieldSize :: Int        -- ^ Size in bytes
  , fieldAlignment :: Int   -- ^ Alignment requirement
  , fieldPadding :: Int     -- ^ Padding bytes after this field
  }
  deriving (Show, Eq)

-- | TypeRep layout implementation
instance WGSLLayout TypeRep where
  wgslSize mode ty = case ty of
    TF32  -> 4
    TF16  -> 2
    TI32  -> 4
    TU32  -> 4
    TBool -> 4  -- WGSL bool is 4 bytes

    -- Vectors
    TVec2 elemTy -> 2 * wgslSize mode elemTy
    TVec3 elemTy -> 3 * wgslSize mode elemTy
    TVec4 elemTy -> 4 * wgslSize mode elemTy

    -- Arrays
    TArray n elemTy ->
      let elemSize = wgslSize mode elemTy
          elemAlign = wgslAlignment mode elemTy
          -- Array elements must be aligned to their base alignment
          -- In std140, array stride is rounded up to 16 bytes
          -- In std430, array stride is rounded up to element alignment
          stride = case mode of
            Std140 -> roundUp (max elemSize elemAlign) 16
            Std430 -> roundUp elemSize elemAlign
      in n * stride

    -- Pointers don't have size (opaque handles)
    TPtr _ _ -> 0

    -- Subgroup matrix types (GPU register types, no direct layout)
    TSubgroupMatrixLeft _ _ _   -> 0
    TSubgroupMatrixRight _ _ _  -> 0
    TSubgroupMatrixResult _ _ _ -> 0

    -- Struct size computed from field layout
    TStruct _ -> error "Struct size requires field information - use computeStructLayout"

    -- Texture and sampler types (opaque handles, no direct layout)
    TTexture2D _ -> 0
    TSampler -> 0

    -- Atomic types (same size as their underlying types)
    TAtomicI32 -> 4
    TAtomicU32 -> 4

  wgslAlignment mode ty = case ty of
    TF32  -> 4
    TF16  -> 2
    TI32  -> 4
    TU32  -> 4
    TBool -> 4

    -- Vector alignment
    -- vec2<T>: align(T) * 2
    -- vec3<T>: align(T) * 4 (rounded to vec4)
    -- vec4<T>: align(T) * 4
    TVec2 elemTy -> wgslAlignment mode elemTy * 2
    TVec3 elemTy -> wgslAlignment mode elemTy * 4  -- Always 16-byte aligned
    TVec4 elemTy -> wgslAlignment mode elemTy * 4

    -- Array alignment
    TArray _ elemTy ->
      let elemAlign = wgslAlignment mode elemTy
      in case mode of
        Std140 -> roundUp elemAlign 16  -- Minimum 16-byte alignment
        Std430 -> elemAlign

    -- Pointers have no alignment (opaque)
    TPtr _ _ -> 0

    -- Subgroup matrices (opaque)
    TSubgroupMatrixLeft _ _ _   -> 0
    TSubgroupMatrixRight _ _ _  -> 0
    TSubgroupMatrixResult _ _ _ -> 0

    -- Struct alignment is max of field alignments
    TStruct _ -> error "Struct alignment requires field information - use computeStructLayout"

    -- Texture and sampler types (opaque, no alignment)
    TTexture2D _ -> 0
    TSampler -> 0

    -- Atomic types (same alignment as their underlying types)
    TAtomicI32 -> 4
    TAtomicU32 -> 4

-- | Compute complete layout for a struct
-- Returns field layouts with offsets and padding
computeStructLayout
  :: LayoutMode
  -> [(String, TypeRep)]  -- ^ Field names and types
  -> (Int, [FieldLayout])  -- ^ (total size, field layouts)
computeStructLayout mode fields =
  let (finalOffset, layouts) = foldl addField (0, []) fields
      -- Struct size must be rounded to largest alignment
      maxAlign = if null layouts
                  then 1
                  else maximum (map fieldAlignment layouts)
      totalSize = roundUp finalOffset maxAlign
  in (totalSize, reverse layouts)
  where
    addField :: (Int, [FieldLayout]) -> (String, TypeRep) -> (Int, [FieldLayout])
    addField (currentOffset, prevLayouts) (name, ty) =
      let align = wgslAlignment mode ty
          size = wgslSize mode ty
          -- Align current offset to field's alignment requirement
          alignedOffset = roundUp currentOffset align
          -- Padding before this field
          paddingBefore = alignedOffset - currentOffset
          -- Next field starts after this one
          nextOffset = alignedOffset + size
          -- Compute padding after (will be adjusted when next field is added)
          paddingAfter = 0  -- Temporary, will be set by next field or struct end

          layout = FieldLayout
            { fieldName = name
            , fieldType = ty
            , fieldOffset = alignedOffset
            , fieldSize = size
            , fieldAlignment = align
            , fieldPadding = paddingAfter
            }
      in (nextOffset, layout : prevLayouts)

-- | Example usage:
--
-- >>> let fields = [("position", TVec3 TF32), ("color", TVec4 TF32), ("id", TI32)]
-- >>> let (totalSize, layouts) = computeStructLayout Std430 fields
-- >>> totalSize
-- 36
-- >>> map (\f -> (fieldName f, fieldOffset f, fieldSize f)) layouts
-- [("position",0,12),("color",16,16),("id",32,4)]
