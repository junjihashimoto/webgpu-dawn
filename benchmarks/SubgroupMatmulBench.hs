{-# LANGUAGE DataKinds #-}
{-# LANGUAGE NumericUnderscores #-}

{-|
Subgroup Matrix Multiplication Benchmark - Optimized MatMul Performance

This benchmark uses the optimized subgroup matmul shader from MatmulSubgroupDSL.hs
with the profiling infrastructure for accurate performance measurement.

Features:
- Optimized subgroup matrix operations (chromium_experimental_subgroup_matrix)
- F16 precision for performance
- Tiled computation (TM=4, TN=2)
- Chrome Tracing export
- Roofline analysis

Usage:
  cabal run bench-subgroup-matmul                    # Run benchmark
  cabal run bench-subgroup-matmul -- --trace         # Export trace.json
  cabal run bench-subgroup-matmul -- --size 2048     # Custom size
  cabal run bench-subgroup-matmul -- --iters 100     # Custom iterations
-}

module Main where

import Prelude (Int, Float, Double, String, Bool(..), Maybe(..), Show, IO, ($), (++), (+), (*), (-), (/), show, div, mod, abs, sum, fromIntegral, putStrLn, zip, read, take, floor, zipWith, maximum, minimum, length, return)
import qualified Prelude as P

import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.Types (Shape(..), NumType(..), Half(..), AnyTensor(..))
import Graphics.WebGPU.Dawn.Tensor (createTensorWithDataPacked)
import Graphics.WebGPU.Dawn.Float16 (f32VecToF16Vec, f16VecToF32Vec)
import WGSL.DSL
import WGSL.Execute (executeShaderNamed)
import WGSL.Profile
import WGSL.Profile.Trace
import WGSL.Analyze
import qualified Data.Vector.Storable as V
import System.Environment (getArgs)
import System.Clock (Clock(..), getTime, diffTimeSpec, toNanoSecs)
import Control.Monad (when, replicateM)
import Text.Printf (printf)

-- | Benchmark configuration
data BenchConfig = BenchConfig
  { benchSize :: !Int         -- ^ Matrix size (NxN)
  , benchIters :: !Int        -- ^ Number of iterations
  , benchTrace :: !Bool       -- ^ Export trace.json
  , benchValidate :: !Bool    -- ^ Validate correctness
  } deriving (Show)

defaultConfig :: BenchConfig
defaultConfig = BenchConfig
  { benchSize = 4096
  , benchIters = 50
  , benchTrace = False
  , benchValidate = True
  }

-- | Parse command line arguments
parseArgs :: [String] -> BenchConfig
parseArgs = go defaultConfig
  where
    go cfg [] = cfg
    go cfg ("--trace":rest) = go (cfg { benchTrace = True }) rest
    go cfg ("--no-validate":rest) = go (cfg { benchValidate = False }) rest
    go cfg ("--size":sizeStr:rest) = go (cfg { benchSize = read sizeStr }) rest
    go cfg ("--iters":iterStr:rest) = go (cfg { benchIters = read iterStr }) rest
    go cfg (_:rest) = go cfg rest

-- | Subgroup matmul shader (from MatmulSubgroupDSL.hs)
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

  -- Calculate base offsets
  let rowStart = wgX * fromIntegral (8 * tm)
      colStart = (wgY * 2 + localY) * fromIntegral (8 * tn)
      baseA = rowStart * fromIntegral k
      baseB = colStart
      cBase = rowStart * fromIntegral n + colStart

  -- Declare subgroup matrices
  axVars <- replicateM tm $ newSubgroupMatrix LeftMatrix TF16 8 8
  bxVars <- replicateM tn $ newSubgroupMatrix RightMatrix TF16 8 8

  -- Create 2D accumulator array initialized to zero
  accVars <- replicateM tn $ replicateM tm $
               newSubgroupMatrixZero ResultMatrix TF16 8 8

  -- HOAS-style loop
  loop 0 (fromIntegral k) 8 $ \kk -> do
    barrier

    let kkU = u32 kk

    -- Load A matrices
    staticFor (zip [0..] axVars) $ \(i, axVar) -> do
      let offset = baseA + kkU + fromIntegral (8 * k * i)
      loadMatrix axVar matA offset (fromIntegral k) (TSubgroupMatrixLeft TF16 8 8)

    -- Load B matrices
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

-- | CPU reference implementation
cpuMatmul :: V.Vector Float -> V.Vector Float -> Int -> Int -> Int -> V.Vector Float
cpuMatmul matA matB m k n = V.generate (m * n) $ \idx ->
  let row = idx `div` n
      col = idx `mod` n
      dotProd = sum [ (matA V.! (row * k + i)) * (matB V.! (col * k + i)) | i <- [0..k - 1] ]
  in dotProd

-- | Validate correctness
validateMatMul :: V.Vector Float -> V.Vector Float -> V.Vector Float -> Int -> Int -> IO ()
validateMatMul matA matB gpuResult m n = do
  putStrLn "[Validate] Computing CPU reference on small sample..."
  let cpuResult = cpuMatmul matA matB m m n
      tolerance = 0.15  -- F16 precision tolerance
      diffs = V.zipWith (\a b -> abs (a - b)) cpuResult gpuResult
      maxDiff = V.maximum diffs
      allClose = V.all (P.<= tolerance) diffs

  if allClose
    then putStrLn $ "[Validate] PASSED: Max diff = " ++ show maxDiff
    else do
      let maxIdx = V.maxIndex diffs
      putStrLn $ "[Validate] FAILED: Max diff = " ++ show maxDiff ++ " at index " ++ show maxIdx
      printf "  CPU: %.6f, GPU: %.6f\n" (cpuResult V.! maxIdx) (gpuResult V.! maxIdx)

main :: IO ()
main = do
  args <- getArgs
  let config = parseArgs args

  putStrLn "=== Subgroup Matrix Multiplication Benchmark (Optimized) ==="
  putStrLn $ "[Config] Size: " ++ show (benchSize config) ++
             ", Iterations: " ++ show (benchIters config)
  putStrLn $ "[Config] Trace: " ++ show (benchTrace config) ++
             ", Validate: " ++ show (benchValidate config)
  putStrLn "[Config] Using: Subgroup ops, F16 precision, TM=4, TN=2"
  putStrLn ""

  -- Create GPU context with subgroup features
  ctx <- if benchTrace config
         then createContextWithFeatures
                ["allow_unsafe_apis"]
                [FeatureTimestampQuery, FeatureShaderF16, FeatureSubgroups,
                 FeatureChromiumExperimentalSubgroupMatrix]
         else createContextWithFeatures
                ["allow_unsafe_apis"]
                [FeatureShaderF16, FeatureSubgroups,
                 FeatureChromiumExperimentalSubgroupMatrix]

  -- Run benchmark
  (times, events) <- runSubgroupMatMulBench ctx config

  -- Print statistics
  printStats config times

  -- Export trace if requested
  when (benchTrace config) $ do
    putStrLn ""
    putStrLn "[Trace] Exporting to trace.json..."
    writeChromeTrace "trace.json" events
    putStrLn "[Trace] Load trace.json in ui.perfetto.dev or chrome://tracing"

  destroyContext ctx

-- | Run subgroup matmul benchmark
runSubgroupMatMulBench :: Context -> BenchConfig -> IO ([Double], [ProfileEvent])
runSubgroupMatMulBench ctx config = do
  let n = benchSize config
      iters = benchIters config
      m = n
      k = n
      tm = 4  -- Tile configuration
      tn = 2
      lid0 = 32
      lid1 = 2

  -- Create input matrices (F32 for CPU validation)
  let matA = V.generate (m * k) (\i -> fromIntegral (i `mod` 100) / 100.0 :: Float)
      matBOrig = V.generate (n * k) (\i -> fromIntegral ((i + 5) `mod` 100) / 100.0 :: Float)
      -- Transpose B for kernel
      matB = V.generate (n * k) (\i ->
        let row = i `div` n
            col = i `mod` n
        in matBOrig V.! (col * k + row)) :: V.Vector Float

  -- Convert to F16
  let matA_w16 = f32VecToF16Vec matA
      matB_w16 = f32VecToF16Vec matB
      matA_f16 = V.map Half matA_w16
      matB_f16 = V.map Half matB_w16

  -- Create GPU tensors
  tensorA <- createTensorWithDataPacked ctx (Shape [m * k]) F16 matA_f16
  tensorB <- createTensorWithDataPacked ctx (Shape [n * k]) F16 matB_f16
  tensorC <- createTensor ctx (Shape [m * n]) F16

  -- Build shader
  let shaderModule = buildShaderWithAutoBinding
        (lid0, lid1, 1)
        (subgroupMatmulShaderDSL m k n tm tn)
      shaderModuleWithExt = shaderModule {
        moduleExtensions = ["f16", "chromium_experimental_subgroup_matrix"],
        moduleDiagnostics = ["off, chromium.subgroup_matrix_uniformity"]
      }
      numWorkgroupsX = (m + 8 * tm - 1) `div` (8 * tm)
      numWorkgroupsY = (n + 8 * tn * lid1 - 1) `div` (8 * tn * lid1)
      wgSize = WorkgroupSize numWorkgroupsX numWorkgroupsY 1

  -- Validate if requested
  when (benchValidate config) $ do
    putStrLn "[Validate] Running validation on small sample..."
    executeShaderNamed ctx shaderModuleWithExt
      [ ("A", AnyTensor tensorA)
      , ("B", AnyTensor tensorB)
      , ("C", AnyTensor tensorC)
      ] wgSize

    gpuResult_f16 <- fromGPU ctx tensorC (m * n) :: IO (V.Vector Half)
    let gpuResult_w16 = V.map (\(Half w) -> w) gpuResult_f16
        gpuResult = f16VecToF32Vec gpuResult_w16
    validateMatMul matA matBOrig gpuResult m n
    putStrLn ""

  -- Warm-up run
  putStrLn "[Warmup] Running 1 iteration..."
  executeShaderNamed ctx shaderModuleWithExt
    [ ("A", AnyTensor tensorA)
    , ("B", AnyTensor tensorB)
    , ("C", AnyTensor tensorC)
    ] wgSize

  -- Benchmark runs
  putStrLn $ "[Bench] Running " ++ show iters ++ " iterations..."
  times <- V.generateM iters $ \i -> do
    when (i `P.mod` 10 P.== 0) $
      putStrLn $ "  Iteration " ++ show i ++ "/" ++ show iters

    start <- getTime Monotonic
    executeShaderNamed ctx shaderModuleWithExt
      [ ("A", AnyTensor tensorA)
      , ("B", AnyTensor tensorB)
      , ("C", AnyTensor tensorC)
      ] wgSize
    end <- getTime Monotonic

    let diff = diffTimeSpec end start
        timeMs = fromIntegral (toNanoSecs diff) / 1_000_000.0
    return timeMs

  -- Collect trace events if enabled
  events <- if benchTrace config
            then collectTraceEvents (V.toList times) n
            else return []

  -- Cleanup
  destroyTensor tensorA
  destroyTensor tensorB
  destroyTensor tensorC

  return (V.toList times, events)

-- | Collect trace events
collectTraceEvents :: [Double] -> Int -> IO [ProfileEvent]
collectTraceEvents times n = do
  let startTime = 0
      events = zipWith (\i timeMs ->
        let start = startTime + sum (take i times) * 1_000_000
            end = start + timeMs * 1_000_000
        in ProfileEvent
             { eventName_ = "SubgroupMatMul_" ++ show n
             , startTime = floor start
             , endTime = floor end
             , eventDurationMs = timeMs
             }
        ) [0..] times
  return events

-- | Print benchmark statistics with Roofline analysis
printStats :: BenchConfig -> [Double] -> IO ()
printStats config times = do
  let n = benchSize config
      avgTime = sum times / fromIntegral (length times)
      minTime = minimum times
      maxTime = maximum times

      -- Calculate TFLOPS (2 * N^3 for MatMul)
      totalFlops = 2.0 * fromIntegral n * fromIntegral n * fromIntegral n :: Double
      tflops = totalFlops / (avgTime / 1000.0) / 1e12

  putStrLn ""
  putStrLn "=== Results ==="
  printf "[Bench] Size: %d, Iterations: %d\n" n (benchIters config)
  printf "[Time]  Avg: %.2f ms | Min: %.2f ms | Max: %.2f ms\n" avgTime minTime maxTime
  printf "[Perf]  %.3f TFLOPS\n" tflops
  putStrLn ""

  -- Roofline static analysis (using F16)
  let analysis = analyzeMatMul n F16
  putStrLn $ rooflineReport analysis
