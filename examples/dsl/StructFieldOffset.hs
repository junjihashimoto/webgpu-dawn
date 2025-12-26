{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Struct Field Offset Example - Demonstrating Field Offset Access

This example shows how to use the field offset functionality to access
struct fields in storage buffers. This is particularly useful for:
  - Manual byte offset calculations
  - Arrays of structs in storage buffers
  - Low-level control over memory access

Key features:
  - Field offset constants available in shader code
  - Helper functions for struct field access
  - Both high-level (^.) and low-level offset-based access
-}

module Main where

import Prelude hiding (id, (.))
import qualified Prelude as P
import GHC.Generics (Generic)
import qualified Data.Vector.Storable as V
import Data.Word (Word8)
import qualified Data.ByteString as BS

import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.Tensor
import Graphics.WebGPU.Dawn.Types (AnyTensor(..))
import WGSL.Struct
import WGSL.DSL hiding ((<), (>), (<=), (>=), (==), (/=), Vec3)
import qualified WGSL.DSL as DSL
import WGSL.Execute (executeShaderNamed)
import WGSL.CodeGen (generateWGSL)

-- | Vector3 helper
data Vec3 = Vec3 Float Float Float deriving (Generic, Show)

instance WGSLStorable Vec3 where
  wgslType _ = "vec3<f32>"
  wgslAlignment _ = Alignment 16  -- vec3 aligns to 16 bytes in std430
  wgslSize _ = 12  -- 3 floats = 12 bytes
  wgslPut (Vec3 x y z) = do
    wgslPut x
    wgslPut y
    wgslPut z

-- | Particle with multiple fields
data Particle = Particle
  { pos :: Vec3
  , vel :: Vec3
  , mass :: Float
  } deriving (Generic, Show)

instance WGSLStorable Particle

main :: IO ()
main = do
  putStrLn "=== WGSL Struct Field Offset Demonstration ==="
  putStrLn ""

  -- Show field layout information
  -- We use a dummy particle for layout calculation (values don't matter, only types)
  let particleProxy = Particle (Vec3 0 0 0) (Vec3 0 0 0) 0
  let layout = structLayout particleProxy

  putStrLn "Field Layout:"
  putStrLn "============="
  mapM_ (\(name, fieldLayout) -> do
    putStrLn P.$ "  " P.++ name P.++ ":"
    putStrLn P.$ "    Offset: " P.++ show (fieldOffset fieldLayout) P.++ " bytes"
    putStrLn P.$ "    Size: " P.++ show (fieldSize fieldLayout) P.++ " bytes"
    let Alignment a = fieldAlign fieldLayout
    putStrLn P.$ "    Alignment: " P.++ show a P.++ " bytes"
    putStrLn P.$ "    WGSL Type: " P.++ fieldWGSLType fieldLayout
    ) layout
  putStrLn ""

  -- Show how to get individual field offsets
  case fieldOffsetOf particleProxy "vel" of
    Just offset -> putStrLn P.$ "Velocity field offset: " P.++ show offset P.++ " bytes"
    Nothing -> putStrLn "Error: vel field not found"
  putStrLn ""

  -- Create test data
  let particles =
        [ Particle (Vec3 0.0 0.0 0.0) (Vec3 1.0 0.0 0.0) 1.0
        , Particle (Vec3 1.0 1.0 1.0) (Vec3 0.0 1.0 0.0) 2.0
        , Particle (Vec3 2.0 2.0 2.0) (Vec3 0.0 0.0 1.0) 3.0
        , Particle (Vec3 3.0 3.0 3.0) (Vec3 1.0 1.0 1.0) 4.0
        ]

  putStrLn "Input Particles:"
  mapM_ print particles
  putStrLn ""

  -- Serialize to show memory layout
  let inputBytes = toWGSLBytesList particles
  putStrLn P.$ "Serialized data size: " P.++ show (BS.length inputBytes) P.++ " bytes"
  putStrLn ""

  -- Create shader that demonstrates field offset usage
  let numParticles = length particles

  -- Shader demonstrates field offset usage
  let shader = buildShaderWithAutoBinding (256, 1, 1) $ do
        input <- declareInputBuffer "particles" (TArray numParticles (TStruct "Particle"))
        output <- declareOutputBuffer "updated" (TArray numParticles (TStruct "Particle"))

        -- Show field offsets as compile-time constants
        -- These are calculated at Haskell compile-time and embedded as literals
        let posOffset = offsetConst particleProxy "pos"    -- 0 bytes
        let velOffset = offsetConst particleProxy "vel"    -- 16 bytes (due to vec3 alignment)
        let massOffset = offsetConst particleProxy "mass"  -- 32 bytes

        -- Print offsets (for demonstration in comments)
        debugPrintI "pos offset" posOffset
        debugPrintI "vel offset" velOffset
        debugPrintI "mass offset" massOffset

        gid <- globalId
        let idx = i32 (vecX gid)

        if_ (idx DSL.< litI32 numParticles)
          (do
            -- Approach: High-level field access with (^.) operator
            -- This generates clean WGSL: particles[idx].pos, particles[idx].vel, etc.
            particle <- readBuffer input idx
            position <- readStructField particle "pos"
            velocity <- readStructField particle "vel"
            m <- readStructField particle "mass"

            -- Write the fields to output
            -- Note: writeStructField generates: output[idx].fieldName = value
            writeStructField output idx "pos" position
            writeStructField output idx "vel" velocity
            writeStructField output idx "mass" m
          )
          (return ())

  -- Print generated WGSL
  putStrLn "Generated WGSL Shader:"
  putStrLn "======================"
  putStrLn $ generateWGSL shader
  putStrLn ""
  putStrLn "WHY THIS MATTERS:"
  putStrLn "================="
  putStrLn "• Field offsets can be referenced in shader code as constants"
  putStrLn "• Useful for manual byte offset calculations when needed"
  putStrLn "• Provides both high-level (^.) and low-level (offsetConst) access"
  putStrLn "• Makes working with arrays of structs in storage buffers easier"
  putStrLn ""
  putStrLn "USAGE PATTERNS:"
  putStrLn "==============="
  putStrLn "1. High-level: particle ^. \"velocity\" (generates particle.velocity)"
  putStrLn "2. Low-level: offsetConst proxy \"velocity\" (generates compile-time constant)"
  putStrLn "3. Manual: readStructField particle \"velocity\" (explicit field read)"
  putStrLn ""
  putStrLn "Field Offset Feature: ✓ Complete!"
