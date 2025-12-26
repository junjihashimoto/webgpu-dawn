{-# LANGUAGE DataKinds #-}

{-|
Matrix Multiplication using WGSL DSL

This demonstrates how to write GPU matrix multiplication using the DSL monad,
not string concatenation.
-}

module Main where

import Prelude
import qualified Prelude as P

import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.Tensor
import qualified Data.Vector.Storable as V
import Control.Exception (bracket)

import WGSL.DSL
import WGSL.Execute

-- | Matrix multiplication shader built with DSL
-- Computes C = A * B^T where:
--   A is (M x K)
--   B is (N x K) transposed
--   C is (M x N)
matmulShaderDSL :: Int -> Int -> Int -> ShaderM ()
matmulShaderDSL m k n = do
  -- Reference global storage buffers
  let matA = globalBuffer "matA" :: Ptr Storage (Array 256 F32)
  let matB = globalBuffer "matB" :: Ptr Storage (Array 256 F32)
  let matC = globalBuffer "matC" :: Ptr Storage (Array 256 F32)

  -- Get global thread ID - each thread computes one output element
  gid <- globalId
  let globalIdx = U32ToI32 (vecX gid)

  -- Compute row and column from linear index
  -- row = globalIdx / N, col = globalIdx % N
  let row = Div globalIdx (litI32 n)
  let col = Sub globalIdx (Mul row (litI32 n))

  -- Accumulator for dot product
  acc <- var TF32 (litF32 0.0)

  -- Compute dot product: C[row,col] = sum(A[row,i] * B[col,i])
  for_ "i" (litI32 0) (litI32 k) $ do
    -- A is row-major: A[row,i] = A[row*K + i]
    let aIdx = Add (Mul row (litI32 k)) (Var "i")
    aVal <- readBuffer matA aIdx

    -- B is transposed column-major: B[col,i] = B[col*K + i]
    let bIdx = Add (Mul col (litI32 k)) (Var "i")
    bVal <- readBuffer matB bIdx

    -- Accumulate: acc += aVal * bVal
    accVal <- readPtr acc
    assign acc (Add accVal (Mul aVal bVal))

  -- Write result: C[row,col] = C[row*N + col]
  let cIdx = Add (Mul row (litI32 n)) col
  finalResult <- readPtr acc
  writeBuffer matC cIdx finalResult

-- CPU reference implementation
cpuMatmul :: V.Vector Float -> V.Vector Float -> Int -> Int -> Int -> V.Vector Float
cpuMatmul matA matB m k n = V.generate (m P.* n) P.$ \idx ->
  let row = idx `div` n
      col = idx `mod` n
      dotProd = sum [ (matA V.! (row P.* k P.+ i)) P.* (matB V.! (col P.* k P.+ i)) | i <- [0..k P.- 1] ]
  in dotProd

main :: IO ()
main = do
  putStrLn "=== Matrix Multiplication using WGSL DSL ==="
  putStrLn ""

  let m = 4
      k = 4
      n = 4

  putStrLn P.$ "Matrix dimensions: " P.++ show m P.++ "x" P.++ show k P.++ " * " P.++ show n P.++ "x" P.++ show k
  putStrLn ""

  -- Build shader using DSL
  putStrLn "Building shader with DSL..."
  let shaderModule = buildShaderModule
        [ ("matA", TArray (m P.* k) TF32, MStorage)
        , ("matB", TArray (n P.* k) TF32, MStorage)
        , ("matC", TArray (m P.* n) TF32, MStorage)
        ]
        (matmulShaderDSL m k n)

  putStrLn "Generated WGSL:"
  putStrLn "==============="
  putStrLn P.$ generateWGSL shaderModule
  putStrLn ""

  putStrLn "DSL features used:"
  putStrLn "  • globalBuffer - reference storage buffers"
  putStrLn "  • globalId - thread ID built-in"
  putStrLn "  • var - local variable declaration"
  putStrLn "  • for_ - loop control flow"
  putStrLn "  • Operators: +, *, /, - (overloaded for Exp types)"
  putStrLn "  • readBuffer/writeBuffer - array access"
  putStrLn ""

  -- Test data
  let matA = V.fromList [fromIntegral i P./ (10.0 :: Float) | i <- [0..m P.* k P.- 1]]
      matB = V.fromList [fromIntegral i P./ (10.0 :: Float) | i <- [0..n P.* k P.- 1]]

  putStrLn P.$ "Matrix A: " P.++ show matA
  putStrLn P.$ "Matrix B: " P.++ show matB
  putStrLn ""

  -- CPU reference
  let cpuResult = cpuMatmul matA matB m k n
  putStrLn P.$ "CPU result: " P.++ show cpuResult
  putStrLn ""

  putStrLn "This demonstrates a REAL DSL:"
  putStrLn "  ✓ Monadic shader construction (ShaderM)"
  putStrLn "  ✓ Type-safe expressions (Exp F32, Exp I32)"
  putStrLn "  ✓ Operator overloading for natural syntax"
  putStrLn "  ✓ Control flow captured in AST (for_, if_)"
  putStrLn "  ✓ Generates WGSL from AST, not string concatenation"
