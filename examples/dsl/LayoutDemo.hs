{-# LANGUAGE DataKinds #-}

{-|
WGSL Memory Layout Demonstration

This example shows how to calculate WGSL memory layouts for structs,
ensuring correct alignment between CPU and GPU data.

Key Issues Addressed:
1. Vec3 alignment (requires 16-byte alignment, even though size is 12)
2. Struct padding (fields must be aligned correctly)
3. Array stride (elements must be properly spaced)
4. std140 vs std430 differences
-}

module Main where

import Prelude
import WGSL.AST
import WGSL.Layout

main :: IO ()
main = do
  putStrLn "=== WGSL Memory Layout Demonstration ==="
  putStrLn ""

  putStrLn "Example 1: Simple Vertex Struct"
  putStrLn "================================"
  putStrLn ""

  let vertexFields =
        [ ("position", TVec3 TF32)  -- 12 bytes, but needs 16-byte alignment
        , ("color", TVec4 TF32)     -- 16 bytes, 16-byte alignment
        , ("texCoord", TVec2 TF32)  -- 8 bytes, 8-byte alignment
        , ("id", TI32)              -- 4 bytes, 4-byte alignment
        ]

  let (std430Size, std430Layout) = computeStructLayout Std430 vertexFields
  let (std140Size, std140Layout) = computeStructLayout Std140 vertexFields

  putStrLn "struct Vertex {"
  putStrLn "  position: vec3<f32>,"
  putStrLn "  color: vec4<f32>,"
  putStrLn "  texCoord: vec2<f32>,"
  putStrLn "  id: i32,"
  putStrLn "}"
  putStrLn ""

  putStrLn "std430 (Storage Buffer Layout):"
  putStrLn $ "Total size: " ++ show std430Size ++ " bytes"
  mapM_ printFieldLayout std430Layout
  putStrLn ""

  putStrLn "std140 (Uniform Buffer Layout):"
  putStrLn $ "Total size: " ++ show std140Size ++ " bytes"
  mapM_ printFieldLayout std140Layout
  putStrLn ""

  putStrLn "Example 2: Vec3 Alignment Issue"
  putStrLn "================================"
  putStrLn ""
  putStrLn "Common mistake: treating vec3<f32> as 12 bytes"
  putStrLn "Reality: vec3<f32> requires 16-byte alignment!"
  putStrLn ""

  let vec3Size = wgslSize Std430 (TVec3 TF32)
  let vec3Align = wgslAlignment Std430 (TVec3 TF32)
  putStrLn $ "vec3<f32> size: " ++ show vec3Size ++ " bytes (3 * 4)"
  putStrLn $ "vec3<f32> alignment: " ++ show vec3Align ++ " bytes (same as vec4!)"
  putStrLn ""

  let problematicFields =
        [ ("normal", TVec3 TF32)    -- 12 bytes, 16-byte aligned
        , ("intensity", TF32)       -- 4 bytes, 4-byte aligned
        ]

  let (probSize, probLayout) = computeStructLayout Std430 problematicFields
  putStrLn "struct Problematic {"
  putStrLn "  normal: vec3<f32>,    // offset 0, size 12"
  putStrLn "  intensity: f32,       // offset needs alignment!"
  putStrLn "}"
  putStrLn ""
  putStrLn $ "Actual layout (total size: " ++ show probSize ++ " bytes):"
  mapM_ printFieldLayout probLayout
  putStrLn ""
  putStrLn "Note: 4 bytes of padding between normal and intensity!"
  putStrLn ""

  putStrLn "Example 3: Array Stride"
  putStrLn "========================"
  putStrLn ""

  let arrayOfVec3 = TArray 10 (TVec3 TF32)
  let std430ArraySize = wgslSize Std430 arrayOfVec3
  let std140ArraySize = wgslSize Std140 arrayOfVec3

  putStrLn "array<vec3<f32>, 10>"
  putStrLn ""
  putStrLn $ "std430 size: " ++ show std430ArraySize ++ " bytes"
  putStrLn $ "  - Stride per element: " ++ show (std430ArraySize `div` 10) ++ " bytes"
  putStrLn $ "  - (vec3 is 12 bytes, rounded to 16 for alignment)"
  putStrLn ""
  putStrLn $ "std140 size: " ++ show std140ArraySize ++ " bytes"
  putStrLn $ "  - Stride per element: " ++ show (std140ArraySize `div` 10) ++ " bytes"
  putStrLn $ "  - (minimum 16-byte stride in std140)"
  putStrLn ""

  putStrLn "Example 4: Scalar vs Vector Alignment"
  putStrLn "======================================"
  putStrLn ""

  let scalarFields =
        [ ("a", TF32)
        , ("b", TF32)
        , ("c", TF32)
        , ("d", TF32)
        ]

  let (scalarSize, scalarLayout) = computeStructLayout Std430 scalarFields

  putStrLn "struct ScalarData { a: f32, b: f32, c: f32, d: f32 }"
  putStrLn $ "Size: " ++ show scalarSize ++ " bytes (tightly packed)"
  mapM_ printFieldLayout scalarLayout
  putStrLn ""

  let vectorField = [("v", TVec4 TF32)]
  let (vecSize, _) = computeStructLayout Std430 vectorField
  putStrLn "struct VectorData { v: vec4<f32> }"
  putStrLn $ "Size: " ++ show vecSize ++ " bytes (same data, different alignment)"
  putStrLn ""

  putStrLn "Key Takeaways:"
  putStrLn "=============="
  putStrLn "1. vec3 requires 16-byte alignment (not 12!)"
  putStrLn "2. Struct fields must be aligned to their natural alignment"
  putStrLn "3. Arrays may need extra stride for alignment"
  putStrLn "4. std140 is more strict than std430"
  putStrLn "5. Use WGSL.Layout to automatically compute correct offsets"
  putStrLn ""

printFieldLayout :: FieldLayout -> IO ()
printFieldLayout f = do
  let typeStr = show (fieldType f)
  putStrLn $ "  " ++ fieldName f ++ ": " ++ typeStr
  putStrLn $ "    offset: " ++ show (fieldOffset f) ++ " bytes"
  putStrLn $ "    size: " ++ show (fieldSize f) ++ " bytes"
  putStrLn $ "    alignment: " ++ show (fieldAlignment f) ++ " bytes"
