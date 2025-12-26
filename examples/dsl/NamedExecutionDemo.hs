{-# LANGUAGE DataKinds #-}

{-|
Safe Named Execution - DSL5 Safety Demo

This example demonstrates the critical safety improvements from DSL5:
1. Name-based buffer binding (prevents order errors)
2. Runtime validation (missing/extra buffer detection)
3. Heterogeneous tensor support (mix F32, I32, U32)

Compare this to the old unsafe executeShader which relied on fragile list order.
-}

module Main where

import Prelude (Int, IO, (++), show, putStrLn, ($), (<>))
import qualified Prelude as P

import WGSL.DSL

-- | Vector addition shader using auto-binding
-- Demonstrates: inputA (binding 0), inputB (binding 1), output (binding 2)
vectorAddShader :: ShaderM ()
vectorAddShader = do
  -- Auto-assigned @binding(0)
  inputA <- declareInputBuffer "inputA" (TArray 256 TF32)

  -- Auto-assigned @binding(1)
  inputB <- declareInputBuffer "inputB" (TArray 256 TF32)

  -- Auto-assigned @binding(2)
  output <- declareOutputBuffer "output" (TArray 256 TF32)

  gid <- globalId
  let idx = U32ToI32 (vecX gid)

  a <- readBuffer inputA idx
  b <- readBuffer inputB idx
  let result = Add a b
  writeBuffer output idx result

main :: IO ()
main = do
  putStrLn "=== Safe Named Execution Demo - DSL5 ==="
  putStrLn ""

  -- Build shader with auto-binding
  let shader = buildShaderWithAutoBinding (256, 1, 1) vectorAddShader

  putStrLn "Shader binding metadata:"
  putStrLn $ "  " <> show (moduleBindings shader)
  putStrLn ""

  putStrLn "Generated WGSL:"
  putStrLn "==============="
  putStrLn $ generateWGSL shader
  putStrLn ""

  putStrLn "Safety Features Demonstrated:"
  putStrLn "=============================="
  putStrLn ""

  putStrLn "1. NAME-BASED EXECUTION (Order Independent)"
  putStrLn "   OLD (Unsafe):"
  putStrLn "     executeShader ctx shader [tensorA, tensorB, output] wgSize"
  putStrLn "     ❌ If you pass [tensorB, tensorA, output] → SILENT ERROR!"
  putStrLn ""
  putStrLn "   NEW (Safe):"
  putStrLn "     executeShaderNamed ctx shader"
  putStrLn "       [ (\"inputA\", AnyTensor tensorA)"
  putStrLn "       , (\"inputB\", AnyTensor tensorB)"
  putStrLn "       , (\"output\", AnyTensor output)"
  putStrLn "       ] wgSize"
  putStrLn "     ✅ Order doesn't matter! Matched by name!"
  putStrLn ""

  putStrLn "2. RUNTIME VALIDATION"
  putStrLn "   Missing buffer:"
  putStrLn "     executeShaderNamed ctx shader"
  putStrLn "       [ (\"inputA\", AnyTensor tensorA)"
  putStrLn "       , (\"output\", AnyTensor output)"
  putStrLn "       ]  -- Missing \"inputB\"!"
  putStrLn "     ❌ Throws: InvalidDataSize \"Missing buffers: [\\\"inputB\\\"]\""
  putStrLn ""
  putStrLn "   Extra/Wrong name:"
  putStrLn "     executeShaderNamed ctx shader"
  putStrLn "       [ (\"inputA\", AnyTensor tensorA)"
  putStrLn "       , (\"wrongName\", AnyTensor tensorB)"
  putStrLn "       , (\"output\", AnyTensor output)"
  putStrLn "       ]"
  putStrLn "     ❌ Throws: InvalidDataSize \"Unknown buffers: [\\\"wrongName\\\"]\""
  putStrLn ""

  putStrLn "3. HETEROGENEOUS TENSORS (Mix F32, I32, U32)"
  putStrLn "   OLD: All tensors must have same type"
  putStrLn "     executeShader :: ... -> [Tensor dtype] -> ..."
  putStrLn "     ❌ Can't mix F32 and I32!"
  putStrLn ""
  putStrLn "   NEW: Can mix any types"
  putStrLn "     executeShaderNamed ctx shader"
  putStrLn "       [ (\"dataF32\", AnyTensor tensorF32)"
  putStrLn "       , (\"indicesI32\", AnyTensor tensorI32)"
  putStrLn "       , (\"outputU32\", AnyTensor tensorU32)"
  putStrLn "       ]"
  putStrLn "     ✅ Type-safe heterogeneous execution!"
  putStrLn ""

  putStrLn "Key Benefits:"
  putStrLn "  ✓ Prevents binding mismatch bugs (critical safety issue)"
  putStrLn "  ✓ Self-documenting (names show intent)"
  putStrLn "  ✓ Refactoring-safe (reorder DSL buffers without breaking runtime)"
  putStrLn "  ✓ Runtime validation catches errors early"
  putStrLn "  ✓ Supports real-world shaders with mixed types"
  putStrLn ""

  putStrLn "Implementation Details:"
  putStrLn "  1. ShaderModule.moduleBindings: [(\"inputA\", 0), (\"inputB\", 1), (\"output\", 2)]"
  putStrLn "  2. executeShaderNamed looks up binding indices by name"
  putStrLn "  3. Tensors are automatically sorted by binding index"
  putStrLn "  4. Validated before execution (missing/extra checks)"
  putStrLn ""

  putStrLn "Recommendation: Use executeShaderNamed for all new code!"
  putStrLn "  - Backward compatible (old executeShader still works)"
  putStrLn "  - Much safer for production code"
  putStrLn "  - Prevents the #1 cause of shader binding bugs"
