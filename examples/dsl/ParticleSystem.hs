{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Particle System Example - Demonstrating Struct Support

This example shows how to use the WGSLStorable type class to automatically
generate WGSL struct definitions and handle memory layout correctly.

Key features:
  - Automatic WGSL struct generation from Haskell records
  - Correct std430 memory alignment
  - Struct field access with (^.) operator
  - Serialization to GPU-compatible bytes
-}

module Main where

import Prelude hiding (id, (.))
import qualified Prelude as P
import GHC.Generics (Generic)
import Data.Int (Int32)
import Data.Word (Word32)
import qualified Data.ByteString as BS

import WGSL.Struct

-- | Vector3 helper (for demonstration)
data Vec3 = Vec3 Float Float Float deriving (Generic, Show)
instance WGSLStorable Vec3 where
  wgslType _ = "vec3<f32>"
  wgslAlignment _ = Alignment 16  -- vec3 aligns to 16 bytes in std430
  wgslSize _ = 12  -- 3 floats = 12 bytes
  wgslPut (Vec3 x y z) = do
    wgslPut x
    wgslPut y
    wgslPut z

-- | Simple particle with Vec3 fields
data ParticleSimple = ParticleSimple
  { pos :: Vec3
  , vel :: Vec3
  , m :: Float
  } deriving (Generic, Show)

instance WGSLStorable ParticleSimple

main :: IO ()
main = do
  putStrLn "=== WGSL Struct System Demonstration ==="
  putStrLn ""
  putStrLn "This example shows automatic struct generation and memory layout."
  putStrLn ""

  -- Example particle
  let particle = ParticleSimple
        { pos = Vec3 1.0 2.0 3.0
        , vel = Vec3 0.1 0.2 0.3
        , m = 5.0
        }

  putStrLn "Haskell Particle:"
  print particle
  putStrLn ""

  -- Show WGSL type
  putStrLn "WGSL Type:"
  putStrLn P.$ "  " P.++ wgslType particle
  putStrLn ""

  -- Show memory layout
  putStrLn "Memory Layout (std430):"
  putStrLn P.$ "  Size: " P.++ show (wgslSize particle) P.++ " bytes"
  let Alignment align = wgslAlignment particle
  putStrLn P.$ "  Alignment: " P.++ show align P.++ " bytes"
  putStrLn ""

  -- Show struct definition (if available)
  case structDefinition particle of
    Just def -> do
      putStrLn "Generated WGSL Struct Definition:"
      putStrLn "=================================="
      putStrLn def
      putStrLn ""
    Nothing -> putStrLn "No struct definition available\n"

  -- Show field layouts
  putStrLn "Field Layout Details:"
  putStrLn "====================="
  let layouts = structLayout particle
  mapM_ (\(name, layout) -> do
    putStrLn P.$ "  " P.++ name P.++ ":"
    putStrLn P.$ "    Offset: " P.++ show (fieldOffset layout) P.++ " bytes"
    putStrLn P.$ "    Size: " P.++ show (fieldSize layout) P.++ " bytes"
    let Alignment a = fieldAlign layout
    putStrLn P.$ "    Alignment: " P.++ show a P.++ " bytes"
    putStrLn P.$ "    WGSL Type: " P.++ fieldWGSLType layout
    ) layouts
  putStrLn ""

  -- Show serialization
  let bytes = toWGSLBytes particle
  putStrLn "Serialized Bytes (GPU-compatible layout):"
  putStrLn P.$ "  Length: " P.++ show (BS.length bytes) P.++ " bytes"
  putStrLn P.$ "  First 16 bytes: " P.++ show (BS.take 16 bytes)
  putStrLn ""

  putStrLn "WHY THIS MATTERS:"
  putStrLn "================="
  putStrLn "• WGSL has strict alignment rules (std430)"
  putStrLn "• vec3<f32> must be aligned to 16 bytes, not 12!"
  putStrLn "• Automatic padding prevents data corruption"
  putStrLn "• Haskell record → WGSL struct is now automatic"
  putStrLn ""
  putStrLn "NEXT STEPS:"
  putStrLn "==========="
  putStrLn "• Use (^.) operator for field access in shaders"
  putStrLn "• Example: particle ^. \"position\" in shader code"
  putStrLn "• Struct definitions automatically included in shader"
  putStrLn ""
  putStrLn "Phase 7 Priority 1: ✓ Complete!"
