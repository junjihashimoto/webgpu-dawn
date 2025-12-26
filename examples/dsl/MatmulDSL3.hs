{-# LANGUAGE DataKinds #-}

{-|
Matrix Multiplication - DSL3 Production-Ready Features Demo

This example demonstrates ALL the new production-ready features from dsl3.md:
1. Multi-dimensional views (safe array indexing)
2. Kernel configuration integration
3. Debug print support
4. Optimized code generation

Compare with MatmulSubgroupDSL2.hs to see the improvements!
-}

module Main where

import Prelude (Int, Float, IO, ($), (++), show, putStrLn, div)
import qualified Prelude as P

import WGSL.DSL

-- | Simple matrix multiplication using DSL3 features
-- Demonstrates views, kernel config, and debug prints
matmulDSL3 :: Int -> Int -> Int -> ShaderM ()
matmulDSL3 m k n = do
  -- Get buffers
  let matA = globalBuffer "A" :: Ptr Storage (Array 1 F32)
  let matB = globalBuffer "B" :: Ptr Storage (Array 1 F32)
  let matC = globalBuffer "C" :: Ptr Storage (Array 1 F32)

  -- Create 2D views for safe indexing
  let viewA = makeView2D matA m k  -- m rows, k columns
  let viewB = makeView2D matB k n  -- k rows, n columns
  let viewC = makeView2D matC m n  -- m rows, n columns

  -- For demo purposes, use simple indexing
  -- In practice, you'd use thread IDs properly with type conversion
  let row = litI 0
  let col = litI 0

  -- Debug print to show what we're computing
  debugPrintI "Computing element at row" row
  debugPrintI "Computing element at col" col

  -- Accumulator
  acc <- var TF32 (litF 0.0)

  -- Dot product using views (no manual offset calculation!)
  for_ "i" (litI 0) (litI k) $ do
    -- Safe 2D indexing - no more row*k+col calculations!
    -- Views automatically calculate: A[row * k + i]
    a <- readView2D viewA row (Var "i")
    b <- readView2D viewB (Var "i") col

    -- Debug print intermediate values
    debugPrintF "a value" a
    debugPrintF "b value" b

    -- Accumulate
    let prod = a `Mul` b
    oldAcc <- readPtr acc
    assign acc (oldAcc `Add` prod)

  -- Write result using view
  -- Views automatically calculate: C[row * n + col]
  finalResult <- readPtr acc
  writeView2D viewC row col finalResult

  debugPrintF "Final result" finalResult

main :: IO ()
main = do
  putStrLn "=== Matrix Multiplication with DSL3 Production Features ==="
  putStrLn ""

  let m = 64
      k = 64
      n = 64

  putStrLn $ "Matrix dimensions: " ++ show m ++ "x" ++ show k ++ " * " ++ show k ++ "x" ++ show n
  putStrLn ""

  -- Use KernelConfig for integrated configuration
  let config = defaultKernelConfig
        { kernelWorkgroupSize = (16, 16, 1)
        , kernelExtensions = []
        , kernelDiagnostics = []
        , kernelFunctionName = "matmul"
        , kernelGlobals =
            [ ("A", TArray (m P.* k) TF32, MStorage)
            , ("B", TArray (k P.* n) TF32, MStorage)
            , ("C", TArray (m P.* n) TF32, MStorage)
            ]
        }

  putStrLn "Building kernel with integrated configuration..."
  let shaderModule = buildKernel config (matmulDSL3 m k n)

  putStrLn ""
  putStrLn "Generated WGSL (unoptimized):"
  putStrLn "============================="
  putStrLn $ generateWGSL shaderModule
  putStrLn ""

  putStrLn "Generated WGSL (optimized with constant folding):"
  putStrLn "================================================="
  putStrLn $ generateWGSLOptimized shaderModule
  putStrLn ""

  putStrLn "DSL3 Production Features Demonstrated:"
  putStrLn "  ✓ Multi-dimensional views - Safe 2D array indexing"
  putStrLn "  ✓ Kernel configuration - Integrated workgroup size and metadata"
  putStrLn "  ✓ Debug prints - GPU printf-style debugging"
  putStrLn "  ✓ Optimized codegen - Constant folding and simplification"
  putStrLn ""

  putStrLn "Key Improvements:"
  putStrLn "  • No manual offset calculations (viewA[row][col] instead of A[row*k+col])"
  putStrLn "  • Self-contained kernel config (no separate setup)"
  putStrLn "  • Debug output for GPU values (comments in generated code)"
  putStrLn "  • Cleaner generated WGSL through optimization"
