{-# LANGUAGE DataKinds #-}

{-|
Automatic Binding Layout - DSL4 Part B Feature Demo

This example demonstrates automatic binding assignment based on declaration order,
as proposed in dsl4.md Section B.

Features:
1. Automatic @binding(N) assignment in order of buffer declaration
2. No manual binding index management
3. Eliminates binding mismatch errors

Compare this to manual binding configuration - much safer!
-}

module Main where

import Prelude (Int, IO, (++), show, putStrLn, ($), unlines, take, lines)
import qualified Prelude as P

import WGSL.DSL

-- | Vector addition shader with automatic binding assignment
-- Demonstrates how buffers are automatically assigned @binding(0), @binding(1), @binding(2)
vectorAddAutoBinding :: ShaderM ()
vectorAddAutoBinding = do
  -- Automatically assigned @binding(0)
  inputA <- declareInputBuffer "inputA" (TArray 256 TF32)

  -- Automatically assigned @binding(1)
  inputB <- declareInputBuffer "inputB" (TArray 256 TF32)

  -- Automatically assigned @binding(2)
  output <- declareOutputBuffer "output" (TArray 256 TF32)

  -- Get thread ID
  gid <- globalId
  let idx = U32ToI32 (vecX gid)

  -- Read from inputs
  a <- readBuffer inputA idx
  b <- readBuffer inputB idx

  -- Compute and write
  let result = Add a b
  writeBuffer output idx result

-- | Matrix multiplication with automatic binding
-- Shows more complex example with 3 buffers
matmulAutoBinding :: Int -> Int -> Int -> ShaderM ()
matmulAutoBinding m k n = do
  -- Buffers are automatically assigned @binding(0), @binding(1), @binding(2) in order
  matA <- declareInputBuffer "A" (TArray (m P.* k) TF32)
  matB <- declareInputBuffer "B" (TArray (k P.* n) TF32)
  matC <- declareOutputBuffer "C" (TArray (m P.* n) TF32)

  gid <- globalId
  let globalIdx = U32ToI32 (vecX gid)
  let row = Div globalIdx (litI32 n)
  let col = Sub globalIdx (Mul row (litI32 n))

  -- Accumulator
  acc <- var TF32 (litF32 0.0)

  -- Dot product
  for_ "i" (litI32 0) (litI32 k) $ do
    let aIdx = Add (Mul row (litI32 k)) (Var "i")
    let bIdx = Add (Mul (Var "i") (litI32 n)) col

    a <- readBuffer matA aIdx
    b <- readBuffer matB bIdx

    let prod = Mul a b
    oldAcc <- readPtr acc
    assign acc (Add oldAcc prod)

  -- Write result
  let cIdx = Add (Mul row (litI32 n)) col
  finalResult <- readPtr acc
  writeBuffer matC cIdx finalResult

main :: IO ()
main = do
  putStrLn "=== Automatic Binding Layout - DSL4 Part B ==="
  putStrLn ""
  putStrLn "This demonstrates automatic @binding(N) assignment based on declaration order."
  putStrLn ""

  putStrLn "Example 1: Vector Addition"
  putStrLn "==========================="
  putStrLn ""
  putStrLn "Shader code:"
  putStrLn "  inputA <- declareInputBuffer \"inputA\" (TArray 256 TF32)   -- auto @binding(0)"
  putStrLn "  inputB <- declareInputBuffer \"inputB\" (TArray 256 TF32)   -- auto @binding(1)"
  putStrLn "  output <- declareOutputBuffer \"output\" (TArray 256 TF32)  -- auto @binding(2)"
  putStrLn ""

  let vecAddModule = buildShaderWithAutoBinding (256, 1, 1) vectorAddAutoBinding
  putStrLn "Generated WGSL:"
  putStrLn "==============="
  putStrLn $ generateWGSL vecAddModule
  putStrLn ""

  putStrLn "Example 2: Matrix Multiplication"
  putStrLn "================================="
  putStrLn ""
  let m = 64
      k = 64
      n = 64

  putStrLn P.$ "Matrix dimensions: " ++ show m ++ "x" ++ show k ++ " * " ++ show k ++ "x" ++ show n
  putStrLn ""
  putStrLn "Shader code:"
  putStrLn "  matA <- declareInputBuffer \"A\" ...   -- auto @binding(0)"
  putStrLn "  matB <- declareInputBuffer \"B\" ...   -- auto @binding(1)"
  putStrLn "  matC <- declareOutputBuffer \"C\" ...  -- auto @binding(2)"
  putStrLn ""

  let matmulModule = buildShaderWithAutoBinding (16, 16, 1) (matmulAutoBinding m k n)
  putStrLn "Generated WGSL (first 30 lines):"
  putStrLn "================================="
  putStrLn P.$ unlines P.$ P.take 30 P.$ lines P.$ generateWGSL matmulModule
  putStrLn ""

  putStrLn "Key Benefits:"
  putStrLn "  ✓ No manual @binding(N) management"
  putStrLn "  ✓ Binding indices assigned automatically in declaration order"
  putStrLn "  ✓ Eliminates binding mismatch errors"
  putStrLn "  ✓ Easy to reorder buffers - bindings update automatically"
  putStrLn ""

  putStrLn "How it works:"
  putStrLn "  1. declareInputBuffer/declareOutputBuffer track buffers in ShaderM state"
  putStrLn "  2. Buffers are registered in order of declaration"
  putStrLn "  3. buildShaderWithAutoBinding extracts buffers and assigns sequential bindings"
  putStrLn "  4. Code generation creates @binding(0), @binding(1), @binding(2), etc."
  putStrLn ""

  putStrLn "Next Steps (Future Work):"
  putStrLn "  • Name-based buffer matching at runtime"
  putStrLn "  • Extract binding layout info for CPU-side buffer setup"
  putStrLn "  • Support for multiple bind groups"
