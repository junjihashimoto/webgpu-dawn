{-# LANGUAGE ScopedTypeVariables #-}

module Graphics.WebGPU.Dawn.Float16
  ( -- * Explicit conversion functions
    f32ToF16
  , f16ToF32
    -- * Vector conversion functions
  , f32VecToF16Vec
  , f16VecToF32Vec
  ) where

import Data.Word (Word16, Word32)
import Data.Bits (shiftR, shiftL, (.&.), (.|.))
import Unsafe.Coerce (unsafeCoerce)
import Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V

-- | Convert Float32 to Float16 (IEEE 754 half precision)
-- This function explicitly converts FP32 to FP16 format.
-- Use this when you need to convert Float data to FP16 for GPU upload.
f32ToF16 :: Float -> Word16
f32ToF16 f =
  let w32 = unsafeCoerce f :: Word32
      sign = (w32 `shiftR` 16) .&. 0x8000
      exp32 = (w32 `shiftR` 23) .&. 0xFF
      mantissa32 = w32 .&. 0x7FFFFF

      -- Handle special cases
      (exp16, mantissa16)
        | exp32 == 0xFF = (0x1F, if mantissa32 == 0 then 0 else 1) -- Inf or NaN
        | exp32 == 0 = (0, 0) -- Zero or denorm → flush to zero
        | otherwise =
            let exp_adj = fromIntegral exp32 - 127 + 15 :: Int
            in if exp_adj <= 0
               then (0, 0) -- Underflow → flush to zero
               else if exp_adj >= 0x1F
                    then (0x1F, 0) -- Overflow → Inf
                    else (fromIntegral exp_adj, mantissa32 `shiftR` 13)
  in fromIntegral $ sign .|. (exp16 `shiftL` 10) .|. (mantissa16 .&. 0x3FF)

-- | Convert Float16 (Word16) to Float32
-- This function explicitly converts FP16 format to FP32.
-- Use this when you need to convert Word16 FP16 data from GPU to Float.
f16ToF32 :: Word16 -> Float
f16ToF32 h = unsafeCoerce (floatBits :: Word32)
  where
    sign = (h `shiftR` 15) .&. 0x1
    exponent = (h `shiftR` 10) .&. 0x1F
    mantissa = h .&. 0x3FF

    sign32 = (fromIntegral sign :: Word32) `shiftL` 31

    floatBits = if exponent == 0
      then  -- Zero or subnormal
        if mantissa == 0
          then sign32  -- Zero
          else sign32  -- Treat subnormals as zero
      else if exponent == 0x1F
        then  -- Infinity or NaN
          sign32 .|. (0xFF `shiftL` 23) .|. ((fromIntegral mantissa :: Word32) `shiftL` 13)
        else  -- Normalized value
          let exp32 = (fromIntegral exponent - 15 + 127 :: Word32) `shiftL` 23
              mant32 = (fromIntegral mantissa :: Word32) `shiftL` 13
          in sign32 .|. exp32 .|. mant32

-- | Convert a Vector of Float32 to Vector of Float16 (Word16)
-- This is a convenience function for bulk conversion.
f32VecToF16Vec :: Vector Float -> Vector Word16
f32VecToF16Vec = V.map f32ToF16

-- | Convert a Vector of Float16 (Word16) to Vector of Float32
-- This is a convenience function for bulk conversion.
f16VecToF32Vec :: Vector Word16 -> Vector Float
f16VecToF32Vec = V.map f16ToF32
