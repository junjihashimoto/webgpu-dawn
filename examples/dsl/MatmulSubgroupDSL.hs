{-# LANGUAGE DataKinds #-}

{-|
Matrix Multiplication - Comprehensive WGSL Phase 6 DSL Example

This is the primary example demonstrating the complete Phase 6 WGSL DSL.

PHASE 6 DSL FEATURES:
  ✓ Natural math operators (a + b * 2 instead of Add (Mul b 2))
  ✓ HOAS loops (loop 0 10 1 $ \i -> ...) - no string variable names!
  ✓ Type-safe conversions (i32, u32, fromIntegral)
  ✓ Num/Fractional instances for clean arithmetic
  ✓ Automatic variable naming with newSubgroupMatrix
  ✓ High-level operations (loadMatrix, storeMatrix, mma)

SAFETY FEATURES:
  ✓ Automatic @binding(N) assignment (declareInputBuffer/declareOutputBuffer)
  ✓ Name-based buffer binding (prevents order errors)
  ✓ Runtime validation (missing/extra buffer detection)
  ✓ Self-documenting code with named buffers

GPU OPTIMIZATION:
  ✓ Subgroup matrix operations (chromium_experimental_subgroup_matrix)
  ✓ F16 precision for performance
  ✓ Tiled computation for cache efficiency

USAGE:
  - Default: Runs GPU verification (compares CPU vs GPU results)
  - DEMO=1: Shows generated WGSL and explains features
-}

module Main where

import Prelude (Int, Float, Bool(..), Maybe(..), IO, ($), (++), (+), (*), (-), (/), show, div, mod, abs, sum, fromIntegral, putStrLn, zip, lines, unlines)
import qualified Prelude as P

import Graphics.WebGPU.Dawn hiding (createKernelCode, dispatchKernel, createContextWithFeatures, createTensorWithData, createTensor, fromGPU)
import Graphics.WebGPU.Dawn.Types (Shape(..), NumType(..), Half(..), AnyTensor(..))
import Graphics.WebGPU.Dawn.ContT
import Graphics.WebGPU.Dawn.Float16 (f32VecToF16Vec, f16VecToF32Vec)
import qualified Data.Vector.Storable as V
import System.Environment (lookupEnv)
import Text.Printf (printf)
import Control.Monad (replicateM, forM_)

import WGSL.DSL
import WGSL.Execute (executeShaderNamed)

-- | Subgroup matrix multiplication shader using PHASE 6 IMPROVED DSL
-- This version showcases the new DSL features:
--   ✓ Natural math operators (a * 2 + 1 instead of Add (Mul a 2) 1)
--   ✓ HOAS loops (no string variable names!)
--   ✓ Type-safe conversions (u32, i32, fromIntegral)
--   ✓ Num/Fractional instances for clean arithmetic
--   ✓ Automatic variable naming with newSubgroupMatrix
--   ✓ High-level operations (loadMatrix, storeMatrix, mma)
--   ✓ Auto-binding with declareInputBuffer/declareOutputBuffer
subgroupMatmulShaderDSL :: Int -> Int -> Int -> Int -> Int -> ShaderM ()
subgroupMatmulShaderDSL m k n tm tn = do
  -- Declare buffers with automatic @binding assignment
  matA <- declareInputBuffer "A" (TArray (m * k) TF16)
  matB <- declareInputBuffer "B" (TArray (n * k) TF16)
  matC <- declareOutputBuffer "C" (TArray (m * n) TF16)

  -- Get built-in IDs
  wg <- workgroupId
  lid <- localId

  let wgX = vecX wg
      wgY = vecY wg
      localY = vecY lid

  -- Calculate base offsets using NATURAL MATH OPERATORS!
  let rowStart = wgX * fromIntegral (8 * tm)
      colStart = (wgY * 2 + localY) * fromIntegral (8 * tn)

      baseA = rowStart * fromIntegral k
      baseB = colStart
      cBase = rowStart * fromIntegral n + colStart

  -- Declare subgroup matrices with automatic naming
  axVars <- replicateM tm $ newSubgroupMatrix LeftMatrix TF16 8 8
  bxVars <- replicateM tn $ newSubgroupMatrix RightMatrix TF16 8 8

  -- Create 2D accumulator array initialized to zero
  accVars <- replicateM tn $ replicateM tm $
               newSubgroupMatrixZero ResultMatrix TF16 8 8

  -- HOAS-style loop - no string variable names!
  loop 0 (fromIntegral k) 8 $ \kk -> do
    barrier

    -- Natural type conversion using u32 helper
    let kkU = u32 kk

    -- Load A matrices - natural arithmetic!
    staticFor (zip [0..] axVars) $ \(i, axVar) -> do
      let offset = baseA + kkU + fromIntegral (8 * k * i)
      loadMatrix axVar matA offset (fromIntegral k) (TSubgroupMatrixLeft TF16 8 8)

    -- Load B matrices - parentheses ensure correct precedence
    staticFor (zip [0..] bxVars) $ \(i, bxVar) -> do
      let offset = baseB + (kkU * fromIntegral n) + fromIntegral (8 * i)
      loadMatrix bxVar matB offset (fromIntegral n) (TSubgroupMatrixRight TF16 8 8)

    -- Multiply-accumulate
    staticFor (zip bxVars accVars) $ \(bxVar, accRow) ->
      staticFor (zip axVars accRow) $ \(axVar, accVar) -> do
        mma accVar axVar bxVar

  barrier

  -- Store results
  staticFor (zip [0..] accVars) $ \(j, accRow) ->
    staticFor (zip [0..] accRow) $ \(i, accVar) -> do
      let offset = cBase + fromIntegral (i * 8 * n + 8 * j)
      storeMatrix matC offset accVar (fromIntegral n)

-- CPU reference implementation
cpuMatmul :: V.Vector Float -> V.Vector Float -> Int -> Int -> Int -> V.Vector Float
cpuMatmul matA matB m k n = V.generate (m * n) $ \idx ->
  let row = idx `div` n
      col = idx `mod` n
      dotProd = sum [ (matA V.! (row * k + i)) * (matB V.! (col * k + i)) | i <- [0..k - 1] ]
  in dotProd

-- Check if two vectors are close
isClose :: V.Vector Float -> V.Vector Float -> Float -> (Bool, Int, Float, Float)
isClose v1 v2 tolerance =
  let diffs = V.zipWith (\a b -> abs (a - b)) v1 v2
      maxDiff = V.maximum diffs
      maxIdx = V.maxIndex diffs
      allClose = V.all (P.<= tolerance) diffs
  in (allClose, maxIdx, maxDiff, v2 V.! maxIdx)  -- Fixed: return GPU value, not CPU

main :: IO ()
main = do
  demoMode <- lookupEnv "DEMO"
  let isDemo = case demoMode of
        Just "1" -> True
        Just "true" -> True
        _ -> False

  if isDemo
    then runDemoMode
    else runVerificationMode

runDemoMode :: IO ()
runDemoMode = do
  putStrLn "=== Matrix Multiplication - Comprehensive WGSL Phase 6 DSL Demo ==="
  putStrLn ""
  putStrLn "This demo showcases:"
  putStrLn "  • Auto-binding and named execution (safety features)"
  putStrLn "  • High-performance subgroup matrix operations"
  putStrLn "  • Natural operators and HOAS loops"
  putStrLn "  • F16 precision and tiled computation"
  putStrLn ""
  putStrLn "═════════════════════════════════════════════════════════════════"
  putStrLn ""

  -- Small test configuration
  let m = 128
      k = 128
      n = 128
      tm = 4
      tn = 2
      lid0 = 32 :: Int
      lid1 = 2 :: Int

  putStrLn "Key Features:"
  putStrLn "  ✓ Subgroup matrix operations (chromium_experimental_subgroup_matrix)"
  putStrLn "  ✓ High-performance tiled computation"
  putStrLn "  ✓ F16 precision for GPU efficiency"
  putStrLn "  ✓ HOAS loops: loop 0 k 8 $ \\kk -> ..."
  putStrLn "  ✓ Natural arithmetic: baseA + kkU + offset"
  putStrLn ""

  putStrLn $ "Matrix dimensions: " ++ show m ++ "x" ++ show k ++ " * " ++ show n ++ "x" ++ show k
  putStrLn $ "Tile configuration: TM=" ++ show tm ++ ", TN=" ++ show tn
  putStrLn $ "Workgroup size: (" ++ show lid0 ++ ", " ++ show lid1 ++ ", 1)"
  putStrLn ""

  -- Build shader using DSL with subgroup extensions and correct workgroup size
  let shaderModule = buildShaderWithAutoBinding
        (lid0, lid1, 1)  -- Workgroup size: (32, 2, 1)
        (subgroupMatmulShaderDSL m k n tm tn)
        -- Auto-binding will extract buffer declarations and assign sequential @binding indices

  -- Add extensions and diagnostics to the generated module
  let shaderModuleWithExt = shaderModule {
        moduleExtensions = ["f16", "chromium_experimental_subgroup_matrix"],
        moduleDiagnostics = ["off, chromium.subgroup_matrix_uniformity"]
      }

  putStrLn "Generated WGSL from DSL (first 50 lines):"
  putStrLn "-----------------------------------------"
  putStrLn $ unlines $ P.take 50 $ lines $ generateWGSL shaderModuleWithExt
  putStrLn ""

  putStrLn "Shader binding metadata (auto-generated):"
  putStrLn $ "  " ++ show (moduleBindings shaderModule)
  putStrLn ""
  putStrLn "Notice: Buffers A, B, C automatically assigned @binding(0), @binding(1), @binding(2)"
  putStrLn ""

  putStrLn "SAFETY FEATURES IN ACTION:"
  putStrLn "--------------------------"
  putStrLn ""
  putStrLn "✓ Auto-Binding: declareInputBuffer/declareOutputBuffer"
  putStrLn "  No manual @binding indices needed!"
  putStrLn ""
  putStrLn "✓ Named Execution: executeShaderNamed"
  putStrLn "  Buffers matched by name, not position"
  putStrLn "  Runtime validation catches missing/extra buffers"
  putStrLn ""

  putStrLn "═══════════════════════════════════════════════════════"
  putStrLn ""
  putStrLn "PHASE 6 DSL FEATURES SUMMARY:"
  putStrLn "=============================="
  putStrLn ""
  putStrLn "1. Natural Math Operators:"
  putStrLn "   a + b * 2          (not: Add a (Mul b 2))"
  putStrLn ""
  putStrLn "2. HOAS Loops (No String Variables!):"
  putStrLn "   loop 0 10 1 $ \\i -> do"
  putStrLn "     ..."
  putStrLn ""
  putStrLn "3. Type Conversions:"
  putStrLn "   i32 (vecX gid)     (not: U32ToI32 (vecX gid))"
  putStrLn "   u32 kk             (not: I32ToU32 kk)"
  putStrLn ""
  putStrLn "4. Auto-Binding Safety:"
  putStrLn "   declareInputBuffer \"inputA\" ...  → @binding(0)"
  putStrLn "   declareInputBuffer \"inputB\" ...  → @binding(1)"
  putStrLn "   declareOutputBuffer \"output\" ... → @binding(2)"
  putStrLn ""
  putStrLn "5. Named Execution (Runtime Safety):"
  putStrLn "   • Order-independent buffer binding"
  putStrLn "   • Missing buffer detection"
  putStrLn "   • Self-documenting code"
  putStrLn ""
  putStrLn "Run with DEMO=1 to see this output!"
  putStrLn "Run without DEMO=1 to execute GPU verification test."

runVerificationMode :: IO ()
runVerificationMode = evalContT $ do
  liftIO $ putStrLn "=== Verification Mode (F32 for CPU comparison) ==="
  liftIO $ putStrLn ""

  let m = 128
      k = 128
      n = 128
      tm = 4
      tn = 2
      lid0 = 32 :: Int
      lid1 = 2 :: Int

  liftIO $ putStrLn $ "Matrix dimensions: " ++ show m ++ "x" ++ show k ++ " * " ++ show n ++ "x" ++ show k
  liftIO $ putStrLn $ "Tile config: TM=" ++ show tm ++ ", TN=" ++ show tn
  liftIO $ putStrLn ""

  let matA = V.generate (m * k) (\i -> fromIntegral (i `mod` 10) / 10.0) :: V.Vector Float
      matBOrig = V.generate (n * k) (\i -> fromIntegral ((i + 5) `mod` 10) / 10.0) :: V.Vector Float
      -- Transpose B
      matB = V.generate (n * k) (\i ->
        let row = i `div` n
            col = i `mod` n
        in matBOrig V.! (col * k + row)) :: V.Vector Float

  liftIO $ putStrLn "Computing CPU reference..."
  let cpuResult = cpuMatmul matA matBOrig m k n
  liftIO $ putStrLn $ "CPU result: first 5: " ++ show (V.take 5 cpuResult)
  liftIO $ putStrLn ""

  liftIO $ putStrLn "Creating context with subgroup features..."
  ctx <- createContextWithFeatures
    ["allow_unsafe_apis"]
    [FeatureShaderF16, FeatureSubgroups, FeatureChromiumExperimentalSubgroupMatrix]
  liftIO $ putStrLn "✓ Context created"

  liftIO $ putStrLn "Creating GPU tensors..."
  -- Convert F32 data to F16 for GPU
  let matA_w16 = f32VecToF16Vec matA
  let matB_w16 = f32VecToF16Vec matB
  -- Wrap Word16 in Half newtype
  let matA_f16 = V.map Half matA_w16
  let matB_f16 = V.map Half matB_w16
  -- Create tensors with F16 type to match shader expectations
  tensorA <- createTensorWithDataPacked ctx (Shape [m * k]) F16 matA_f16
  tensorB <- createTensorWithDataPacked ctx (Shape [n * k]) F16 matB_f16
  tensorC <- createTensor ctx (Shape [m * n]) F16
  liftIO $ putStrLn "✓ Tensors created"

  liftIO $ putStrLn "Generating shader using IMPROVED DSL with auto-binding..."
  let shaderModule = buildShaderWithAutoBinding
        (lid0, lid1, 1)  -- Workgroup size: (32, 2, 1)
        (subgroupMatmulShaderDSL m k n tm tn)

  -- Add extensions and diagnostics to the generated module
  let shaderModuleWithExt = shaderModule {
        moduleExtensions = ["f16", "chromium_experimental_subgroup_matrix"],
        moduleDiagnostics = ["off, chromium.subgroup_matrix_uniformity"]
      }

  liftIO $ putStrLn "✓ Shader generated from IMPROVED DSL"

  liftIO $ putStrLn "Executing with NAMED buffer binding (safe!)..."
  liftIO $ putStrLn "  Using executeShaderNamed - order independent, type-safe"

  let numWorkgroupsX = (m + 8 * tm - 1) `div` (8 * tm)
      numWorkgroupsY = (n + 8 * tn * lid1 - 1) `div` (8 * tn * lid1)

  -- Use executeShaderNamed for safe, order-independent execution
  liftIO $ executeShaderNamed ctx shaderModuleWithExt
    [ ("A", AnyTensor tensorA)
    , ("B", AnyTensor tensorB)
    , ("C", AnyTensor tensorC)
    ]
    (WorkgroupSize numWorkgroupsX numWorkgroupsY 1)

  liftIO $ putStrLn "✓ Kernel executed with named bindings (safe!)"

  liftIO $ putStrLn "Reading results..."
  -- Read F16 data and convert to F32 for comparison
  gpuResult_f16 <- liftIO $ fromGPU ctx tensorC (m * n) :: ContT r IO (V.Vector Half)
  -- Unwrap Half newtype to Word16 for conversion
  let gpuResult_w16 = V.map (\(Half w) -> w) gpuResult_f16
  let gpuResult = f16VecToF32Vec gpuResult_w16
  liftIO $ putStrLn $ "GPU result: first 5: " ++ show (V.take 5 gpuResult)
  liftIO $ putStrLn ""

  -- F16 has ~11 bits mantissa ≈ 3 decimal digits precision
  -- For values ~20-30, expect errors ~0.01-0.1, so use tolerance of 0.15
  let tolerance = 0.15  -- Appropriate for F16 precision
      (match, maxIdx, maxDiff, gpuVal) = isClose cpuResult gpuResult tolerance

  liftIO $ if match
    then P.putStrLn "✓ VERIFICATION PASSED!"
    else do
      P.putStrLn "✗ VERIFICATION FAILED!"
      printf "  Max diff: %.6f at index %d\n" maxDiff maxIdx
      printf "  CPU: %.6f, GPU: %.6f\n" (cpuResult V.! maxIdx) gpuVal
      printf "  Location: row=%d, col=%d\n" (maxIdx `P.div` n) (maxIdx `P.mod` n)
      -- Show a few values around the error
      P.putStrLn "  Nearby values (index: CPU vs GPU):"
      forM_ [P.max 0 (maxIdx - 2) .. P.min (m * n - 1) (maxIdx + 2)] $ \i ->
        printf "    %d: %.3f vs %.3f (diff=%.3f)\n" i
          (cpuResult V.! i) (gpuResult V.! i)
          (abs ((cpuResult V.! i) - (gpuResult V.! i)))
