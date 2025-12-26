{-# LANGUAGE DataKinds #-}

{-|
Matrix Multiplication using Subgroup Matrix Operations - IMPROVED DSL VERSION

This demonstrates the improved WGSL DSL with:
  • Automatic variable naming (no manual string manipulation)
  • staticFor for compile-time unrolling (clearer intent)
  • Smart constructors hiding internal types
  • High-level operation wrappers (loadMatrix, storeMatrix, mma)
  • Better literal handling

Compare this to MatmulSubgroupDSL.hs to see the improvements!
-}

module Main where

import Prelude (Int, Float, Bool(..), String, Maybe(..), IO, ($), (.), (++), show, div, mod, abs, sum, fromIntegral, putStrLn, zip)
import qualified Prelude as P

import Graphics.WebGPU.Dawn hiding (createKernelCode, dispatchKernel, createContextWithFeatures, createTensorWithData, createTensor, fromGPU)
import Graphics.WebGPU.Dawn.Types (Shape(..), NumType(..), Half(..))
import Graphics.WebGPU.Dawn.ContT
import Graphics.WebGPU.Dawn.Float16 (f32VecToF16Vec, f16VecToF32Vec)
import qualified Data.Vector.Storable as V
import System.Environment (lookupEnv)
import Text.Printf (printf)
import Control.Monad (replicateM, forM_)

import WGSL.DSL

-- | Subgroup matrix multiplication shader using IMPROVED DSL
-- This version is MUCH cleaner thanks to:
--   - newSubgroupMatrix (automatic naming)
--   - staticFor (clear compile-time unrolling)
--   - loadMatrix/storeMatrix/mma (high-level operations)
--   - zip instead of !! (safer indexing)
subgroupMatmulShaderDSL :: Int -> Int -> Int -> Int -> ShaderM ()
subgroupMatmulShaderDSL k n tm tn = do
  -- Reference global storage buffers (type inference)
  let matA = globalBuffer "A" :: Ptr Storage (Array 1 F16)
  let matB = globalBuffer "B" :: Ptr Storage (Array 1 F16)
  let matC = globalBuffer "C" :: Ptr Storage (Array 1 F16)

  -- Get built-in IDs
  wg <- workgroupId
  lid <- localId

  let wgX = vecX wg
  let wgY = vecY wg
  let localY = vecY lid

  -- Calculate base offsets using cleaner literals
  let rowStart = wgX `Mul` litU (8 P.* tm)
  let colStart = (wgY `Mul` litU 2 `Add` localY) `Mul` litU (8 P.* tn)

  let baseA = rowStart `Mul` litU k
  let baseB = colStart
  let cBase = rowStart `Mul` litU n `Add` colStart

  -- Declare subgroup matrices with AUTOMATIC naming
  -- No more "Ax_" ++ show i - the DSL handles it!
  axVars <- replicateM tm $ newSubgroupMatrix LeftMatrix TF16 8 8
  bxVars <- replicateM tn $ newSubgroupMatrix RightMatrix TF16 8 8

  -- Create 2D accumulator array (tn x tm) initialized to zero
  accVars <- replicateM tn $ replicateM tm $
               newSubgroupMatrixZero ResultMatrix TF16 8 8

  -- Main computation loop over K dimension (increment by 8!)
  forStep_ "kk" (litI 0) (litI k) (litI 8) $ do
    barrier

    -- Convert kk from I32 to U32 for arithmetic with workgroup IDs
    let kkU = I32ToU32 (Var "kk" :: Exp I32)

    -- Load A matrices using staticFor with zip (no !! indexing!)
    staticFor (zip [0..] axVars) $ \(i, axVar) -> do
      let offset = baseA `Add` kkU `Add` litU (8 P.* k P.* i)
      loadMatrix axVar matA offset (litU k) (TSubgroupMatrixLeft TF16 8 8)

    -- Load B matrices using staticFor with zip
    staticFor (zip [0..] bxVars) $ \(i, bxVar) -> do
      let offset = baseB `Add` (kkU `Mul` litU n) `Add` litU (8 P.* i)  -- Fixed: parentheses for correct precedence
      loadMatrix bxVar matB offset (litU n) (TSubgroupMatrixRight TF16 8 8)

    -- Multiply-accumulate
    staticFor (zip bxVars accVars) $ \(bxVar, accRow) ->
      staticFor (zip axVars accRow) $ \(axVar, accVar) -> do
        mma accVar axVar bxVar

  barrier

  -- Store results using staticFor with zip
  staticFor (zip [0..] accVars) $ \(j, accRow) ->
    staticFor (zip [0..] accRow) $ \(i, accVar) -> do
      let offset = cBase `Add` litU (i P.* 8 P.* n P.+ 8 P.* j)
      storeMatrix matC offset accVar (litU n)

-- CPU reference implementation
cpuMatmul :: V.Vector Float -> V.Vector Float -> Int -> Int -> Int -> V.Vector Float
cpuMatmul matA matB m k n = V.generate (m P.* n) $ \idx ->
  let row = idx `div` n
      col = idx `mod` n
      dotProd = sum [ (matA V.! (row P.* k P.+ i)) P.* (matB V.! (col P.* k P.+ i)) | i <- [0..k P.- 1] ]
  in dotProd

-- Check if two vectors are close
isClose :: V.Vector Float -> V.Vector Float -> Float -> (Bool, Int, Float, Float)
isClose v1 v2 tolerance =
  let diffs = V.zipWith (\a b -> abs (a P.- b)) v1 v2
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
  putStrLn "=== Matrix Multiplication using IMPROVED Subgroup DSL ==="
  putStrLn ""
  putStrLn "This version uses the IMPROVED DSL with:"
  putStrLn "  • newSubgroupMatrix (automatic naming)"
  putStrLn "  • staticFor (compile-time unrolling)"
  putStrLn "  • loadMatrix/storeMatrix/mma (high-level wrappers)"
  putStrLn "  • zip instead of !! (safer list iteration)"
  putStrLn ""

  -- Small test configuration
  let m = 128
      k = 128
      n = 128
      tm = 4
      tn = 2
      lid0 = 32 :: Int
      lid1 = 2 :: Int

  putStrLn $ "Matrix dimensions: " ++ show m ++ "x" ++ show k ++ " * " ++ show n ++ "x" ++ show k
  putStrLn $ "Tile configuration: TM=" ++ show tm ++ ", TN=" ++ show tn
  putStrLn ""

  -- Build shader using DSL with subgroup extensions and correct workgroup size
  putStrLn "Building shader with IMPROVED DSL..."
  let shaderModule = buildShaderModuleWithConfig
        [ ("A", TArray (m P.* k) TF16, MStorage)
        , ("B", TArray (n P.* k) TF16, MStorage)
        , ("C", TArray (m P.* n) TF16, MStorage)
        ]
        ["f16", "chromium_experimental_subgroup_matrix"]
        ["off, chromium.subgroup_matrix_uniformity"]
        (lid0, lid1, 1)  -- Workgroup size: (32, 2, 1)
        (subgroupMatmulShaderDSL k n tm tn)

  putStrLn "Generated WGSL from DSL:"
  putStrLn "========================"
  putStrLn $ generateWGSL shaderModule
  putStrLn ""

  putStrLn "IMPROVED DSL features demonstrated:"
  putStrLn "  ✓ Automatic variable naming (no string manipulation)"
  putStrLn "  ✓ staticFor for compile-time unrolling"
  putStrLn "  ✓ Smart constructors (newSubgroupMatrix)"
  putStrLn "  ✓ High-level operations (loadMatrix, storeMatrix, mma)"
  putStrLn "  ✓ Safe list iteration with zip (no !!)"
  putStrLn "  ✓ Cleaner literals (litU, litI, litF)"
  putStrLn ""

  putStrLn "Compare this code to MatmulSubgroupDSL.hs to see the improvements!"

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

  let matA = V.generate (m P.* k) (\i -> fromIntegral (i `mod` 10) P./ 10.0) :: V.Vector Float
      matBOrig = V.generate (n P.* k) (\i -> fromIntegral ((i P.+ 5) `mod` 10) P./ 10.0) :: V.Vector Float
      -- Transpose B
      matB = V.generate (n P.* k) (\i ->
        let row = i `div` n
            col = i `mod` n
        in matBOrig V.! (col P.* k P.+ row)) :: V.Vector Float

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
  tensorA <- createTensorWithDataPacked ctx (Shape [m P.* k]) F16 matA_f16
  tensorB <- createTensorWithDataPacked ctx (Shape [n P.* k]) F16 matB_f16
  tensorC <- createTensor ctx (Shape [m P.* n]) F16
  liftIO $ putStrLn "✓ Tensors created"

  liftIO $ putStrLn "Generating shader using IMPROVED DSL..."
  let shaderModule = buildShaderModuleWithConfig
        [ ("A", TArray (m P.* k) TF16, MStorage)
        , ("B", TArray (n P.* k) TF16, MStorage)
        , ("C", TArray (m P.* n) TF16, MStorage)
        ]
        ["f16", "chromium_experimental_subgroup_matrix"]
        ["off, chromium.subgroup_matrix_uniformity"]
        (lid0, lid1, 1)  -- Workgroup size: (32, 2, 1)
        (subgroupMatmulShaderDSL k n tm tn)
  let shaderCode = generateWGSL shaderModule

  liftIO $ putStrLn "✓ Shader generated from IMPROVED DSL"

  kernelCode <- createKernelCode shaderCode
  liftIO $ putStrLn "✓ Shader compiled"

  let numWorkgroupsX = (m P.+ 8 P.* tm P.- 1) `div` (8 P.* tm)
      numWorkgroupsY = (n P.+ 8 P.* tn P.* lid1 P.- 1) `div` (8 P.* tn P.* lid1)

  kernel <- createKernel ctx kernelCode [tensorA, tensorB, tensorC]
            (WorkgroupSize numWorkgroupsX numWorkgroupsY 1)
  liftIO $ putStrLn "✓ Kernel compiled"

  liftIO $ putStrLn "Executing kernel..."
  liftIO $ dispatchKernel ctx kernel
  liftIO $ putStrLn "✓ Kernel completed"

  liftIO $ putStrLn "Reading results..."
  -- Read F16 data and convert to F32 for comparison
  gpuResult_f16 <- liftIO $ fromGPU ctx tensorC (m P.* n) :: ContT r IO (V.Vector Half)
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
      forM_ [P.max 0 (maxIdx P.- 2) .. P.min (m P.* n P.- 1) (maxIdx P.+ 2)] $ \i ->
        printf "    %d: %.3f vs %.3f (diff=%.3f)\n" i
          (cpuResult V.! i) (gpuResult V.! i)
          (abs ((cpuResult V.! i) P.- (gpuResult V.! i)))
