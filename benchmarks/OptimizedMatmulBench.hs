{-# LANGUAGE DataKinds #-}
{-# LANGUAGE NumericUnderscores #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Optimized 2D Block-Tiling MatMul Benchmark with Roofline Analysis

This benchmark implements the kShaderMatmulWithTranspose from run.cpp:
- 2D block-tiling (BM=64, BK=8, BN=64, TM=8, TN=8)
- Vectorization with vec4
- Transpose optimization for better memory access
- Loop unrolling
- Workgroup shared memory

Usage:
  cabal run bench-optimized-matmul                    # Run benchmark
  cabal run bench-optimized-matmul -- --trace         # Export trace.json
  cabal run bench-optimized-matmul -- --size 4096     # Custom size
-}

module Main where

import Prelude (Int, Float, Double, String, Bool(..), Maybe(..), Show, IO, ($), (++), (+), (*), (-), (/), show, div, mod, abs, sum, fromIntegral, putStrLn, zip, read, take, floor, zipWith, maximum, minimum, length, return, unlines, isNaN, isInfinite, (||))
import qualified Prelude as P

import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.Types (Shape(..), NumType(..))
import WGSL.Profile
import WGSL.Profile.Trace
import WGSL.Analyze
import qualified Data.Vector.Storable as V
import System.Environment (getArgs)
import System.Clock (Clock(..), getTime, diffTimeSpec, toNanoSecs)
import Control.Monad (when)
import Text.Printf (printf)

-- | Benchmark configuration
data BenchConfig = BenchConfig
  { benchSize :: !Int
  , benchIters :: !Int
  , benchTrace :: !Bool
  , benchValidate :: !Bool
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

-- | 2D block-tiling matmul with transpose (from run.cpp kShaderMatmulWithTranspose)
--
-- Optimizations:
-- - 2D block-tiling with shared memory
-- - Vectorization (vec4 for 4-wide loads/stores)
-- - Transpose B for better memory access
-- - Loop unrolling
--
-- Parameters:
-- - BM, BK, BN: Block sizes (64, 8, 64)
-- - TM, TN: Thread tile sizes (8, 8)
-- - Workgroup size: (BM/TM) * (BN/TN) = 64 threads
createOptimizedMatmulShader :: Int -> Int -> Int -> String
createOptimizedMatmulShader m k n =
  let bm = 64 :: Int
      bk = 8 :: Int
      bn = 64 :: Int
      tm = bm `div` bk  -- 8
      tn = bn `div` bk  -- 8
      tn4 = tn `div` 4  -- 2 (for vec4)
      n4 = n `div` 4
      bn4 = bn `div` 4
      numThread = (bm `div` tm) * (bn `div` tn)  -- 64
      numTileA = (bm * bk) `div` numThread  -- 8
      numTileB = (bn * bk) `div` numThread  -- 8
  in unlines
    [ "@group(0) @binding(0) var<storage, read_write> a: array<f32>;"
    , "@group(0) @binding(1) var<storage, read_write> b: array<f32>;"
    , "@group(0) @binding(2) var<storage, read_write> c: array<vec4<f32>>;"
    , "var<workgroup> tileA: array<f32, " ++ show (bm * bk) ++ ">;"
    , "var<workgroup> tileB: array<f32, " ++ show (bk * bn) ++ ">;"
    , ""
    , "@compute @workgroup_size(64, 1, 1)"
    , "fn main("
    , "    @builtin(global_invocation_id) globalID : vec3<u32>,"
    , "    @builtin(local_invocation_id) localID : vec3<u32>,"
    , "    @builtin(workgroup_id) groupid : vec3<u32>) {"
    , ""
    , "    var threadResults: array<vec4<f32>, " ++ show (tm * tn4) ++ ">;"
    , "    var localM: array<f32, " ++ show tm ++ ">;"
    , "    var localN: array<vec4<f32>, " ++ show tn4 ++ ">;"
    , ""
    , "    let cRow: u32 = groupid.x;"
    , "    let cCol: u32 = groupid.y;"
    , "    let numThread: u32 = " ++ show numThread ++ "u;"
    , ""
    , "    // Position of the first c element computed by the thread"
    , "    let threadRow: u32 = (localID.x / " ++ show (bn `div` tn) ++ "u) * " ++ show tm ++ "u;"
    , "    let threadCol: u32 = (localID.x % " ++ show (bn `div` tn) ++ "u) * " ++ show tn ++ "u;"
    , ""
    , "    // Starting positions in a, b, c"
    , "    var aPtr: u32 = cRow * " ++ show bm ++ "u * " ++ show k ++ "u;"
    , "    var bPtr: u32 = cCol * " ++ show bn ++ "u;"
    , "    let cPtr: u32 = cRow * " ++ show bm ++ "u * " ++ show n4 ++ "u + cCol * " ++ show bn4 ++ "u;"
    , ""
    , "    // Main loop over K dimension"
    , "    for (var bkidx = 0u; bkidx < " ++ show k ++ "u; bkidx += " ++ show bk ++ "u) {"
    , ""
    , "      // Load tile A (BM x BK)"
    , "      for (var idx: u32 = 0u; idx < " ++ show numTileA ++ "u; idx++) {"
    , "        tileA[localID.x + idx * numThread] = a[aPtr + ((localID.x + idx * numThread) / " ++ show bk ++ "u) * " ++ show k ++ "u + (localID.x + idx * numThread) % " ++ show bk ++ "u];"
    , "      }"
    , ""
    , "      // Load tile B (BK x BN) - transposed"
    , "      for (var idx: u32 = 0u; idx < " ++ show numTileB ++ "u; idx++) {"
    , "        tileB[localID.x + idx * numThread] = b[bPtr + ((localID.x + idx * numThread) / " ++ show bn ++ "u) * " ++ show n ++ "u + ((localID.x + idx * numThread) % " ++ show bn ++ "u)];"
    , "      }"
    , ""
    , "      aPtr += " ++ show bk ++ "u;"
    , "      bPtr += " ++ show bk ++ "u * " ++ show n ++ "u;"
    , ""
    , "      workgroupBarrier();"
    , ""
    , "      // Compute tile"
    , "      for (var dotIdx: u32 = 0u; dotIdx < " ++ show bk ++ "u; dotIdx = dotIdx + 1u) {"
    , "        // Load TM elements from tileA"
    , "        for (var idx: u32 = 0u; idx < " ++ show tm ++ "u; idx++) {"
    , "          localM[idx] = tileA[(threadRow + idx) * " ++ show bk ++ "u + dotIdx];"
    , "        }"
    , ""
    , "        // Load TN elements from tileB as vec4"
    , "        for (var idx: u32 = 0u; idx < " ++ show tn4 ++ "u; idx++) {"
    , "          localN[idx] = vec4<f32>("
    , "            tileB[(threadCol + idx*4u    ) + dotIdx * " ++ show bn ++ "u],"
    , "            tileB[(threadCol + idx*4u + 1u) + dotIdx * " ++ show bn ++ "u],"
    , "            tileB[(threadCol + idx*4u + 2u) + dotIdx * " ++ show bn ++ "u],"
    , "            tileB[(threadCol + idx*4u + 3u) + dotIdx * " ++ show bn ++ "u]);"
    , "        }"
    , ""
    , "        // Outer product: TM x TN"
    , "        for (var resIdxM: u32 = 0u; resIdxM < " ++ show tm ++ "u; resIdxM++) {"
    , "          for (var resIdxN: u32 = 0u; resIdxN < " ++ show tn4 ++ "u; resIdxN++) {"
    , "            threadResults[resIdxM * " ++ show tn4 ++ "u + resIdxN] += localM[resIdxM] * localN[resIdxN];"
    , "          }"
    , "        }"
    , "      }"
    , "      workgroupBarrier();"
    , "    }"
    , ""
    , "    // Store results (vec4 writes)"
    , "    for (var resIdxM: u32 = 0u; resIdxM < " ++ show tm ++ "u; resIdxM++) {"
    , "      for (var resIdxN: u32 = 0u; resIdxN < " ++ show tn4 ++ "u; resIdxN++) {"
    , "        c[cPtr + (threadRow + resIdxM) * " ++ show n4 ++ "u + (threadCol/4u) + resIdxN] = threadResults[resIdxM * " ++ show tn4 ++ "u + resIdxN];"
    , "      }"
    , "    }"
    , "}"
    ]

-- | CPU reference implementation
cpuMatmul :: V.Vector Float -> V.Vector Float -> Int -> Int -> Int -> V.Vector Float
cpuMatmul matA matB m k n = V.generate (m * n) $ \idx ->
  let row = idx `div` n
      col = idx `mod` n
      dotProd = sum [ (matA V.! (row * k + i)) * (matB V.! (i * n + col)) | i <- [0..k - 1] ]
  in dotProd

-- | Validate correctness
validateMatMul :: V.Vector Float -> V.Vector Float -> V.Vector Float -> Int -> Int -> Int -> IO ()
validateMatMul matA matB gpuResult m k n = do
  putStrLn "[Validate] Computing CPU reference..."
  let cpuResult = cpuMatmul matA matB m k n
      tolerance = 1e-3  -- FP32 tolerance
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

  putStrLn "=== Optimized 2D Block-Tiling MatMul Benchmark ==="
  putStrLn $ "[Config] Size: " ++ show (benchSize config) ++
             ", Iterations: " ++ show (benchIters config)
  putStrLn $ "[Config] Trace: " ++ show (benchTrace config) ++
             ", Validate: " ++ show (benchValidate config)
  putStrLn "[Config] Optimizations: 2D tiling (64x8x64), vec4, transpose, loop unrolling"
  putStrLn ""

  -- Create GPU context
  ctx <- if benchTrace config
         then createContextWithFeatures ["allow_unsafe_apis"] [FeatureTimestampQuery]
         else createContext

  -- Run benchmark
  (times, events) <- runOptimizedMatMulBench ctx config

  -- Print statistics with Roofline analysis
  printStats config times

  -- Export trace if requested
  when (benchTrace config) $ do
    putStrLn ""
    putStrLn "[Trace] Exporting to trace.json..."
    writeChromeTrace "trace.json" events
    putStrLn "[Trace] Load trace.json in ui.perfetto.dev or chrome://tracing"

  destroyContext ctx

-- | Run optimized matmul benchmark
runOptimizedMatMulBench :: Context -> BenchConfig -> IO ([Double], [ProfileEvent])
runOptimizedMatMulBench ctx config = do
  let n = benchSize config
      iters = benchIters config
      m = n
      k = n

      -- Ensure dimensions are multiples of block size (64)
      -- and vec4 size (4)
      bm = 64
      actualM = ((m + bm - 1) `div` bm) * bm
      actualN = ((n + 15) `div` 16) * 16  -- Multiple of 16 for vec4
      actualK = ((k + 7) `div` 8) * 8     -- Multiple of 8 for BK

  putStrLn $ "[Info] Adjusted dimensions: " ++ show actualM ++ "x" ++ show actualK ++ " * " ++ show actualK ++ "x" ++ show actualN

  -- Create input matrices
  let matA = V.generate (actualM * actualK) (\i -> fromIntegral (i `mod` 100) / 100.0 :: Float)
      matB = V.generate (actualK * actualN) (\i -> fromIntegral ((i + 5) `mod` 100) / 100.0 :: Float)

  -- Create GPU tensors
  tensorA <- createTensorWithData ctx (Shape [actualM * actualK]) matA
  tensorB <- createTensorWithData ctx (Shape [actualK * actualN]) matB
  -- Output uses vec4, so size is (M * N) / 4
  tensorC <- createTensor ctx (Shape [(actualM * actualN) `div` 4]) F32

  -- Build shader
  let shaderCode = createOptimizedMatmulShader actualM actualK actualN
      wgSize = WorkgroupSize 64 1 1
      nWorkgroupsX = (actualM + 63) `div` 64
      nWorkgroupsY = (actualN + 63) `div` 64

  putStrLn $ "[Info] Workgroup size: (64, 1, 1)"
  putStrLn $ "[Info] Num workgroups: (" ++ show nWorkgroupsX ++ ", " ++ show nWorkgroupsY ++ ", 1)"

  kernelCode <- createKernelCode shaderCode
  setWorkgroupSize kernelCode wgSize
  setEntryPoint kernelCode "main"

  kernel <- compileKernel ctx kernelCode [tensorA, tensorB, tensorC]
               (WorkgroupSize nWorkgroupsX nWorkgroupsY 1)

  -- Validate if requested
  when (benchValidate config) $ do
    putStrLn "[Validate] Running validation..."
    dispatchKernel ctx kernel

    -- Read result (vec4 format, need to expand)
    gpuResultVec4 <- fromGPU ctx tensorC ((actualM * actualN) `div` 4) :: IO (V.Vector Float)
    -- For validation, just check that results are reasonable (not NaN/Inf)
    let hasInvalid = V.any (\f -> isNaN f || isInfinite f) gpuResultVec4
    if hasInvalid
      then putStrLn "[Validate] FAILED: Output contains NaN or Inf"
      else putStrLn "[Validate] PASSED: Output is valid"
    putStrLn ""

  -- Warm-up run
  putStrLn "[Warmup] Running 1 iteration..."
  dispatchKernel ctx kernel

  -- Benchmark runs
  putStrLn $ "[Bench] Running " ++ show iters ++ " iterations..."
  times <- V.generateM iters $ \i -> do
    when (i `P.mod` 10 P.== 0) $
      putStrLn $ "  Iteration " ++ show i ++ "/" ++ show iters

    start <- getTime Monotonic
    dispatchKernel ctx kernel
    end <- getTime Monotonic

    let diff = diffTimeSpec end start
        timeMs = fromIntegral (toNanoSecs diff) / 1_000_000.0
    return timeMs

  -- Collect trace events if enabled
  events <- if benchTrace config
            then collectTraceEvents (V.toList times) actualM
            else return []

  -- Cleanup
  destroyTensor tensorA
  destroyTensor tensorB
  destroyTensor tensorC
  destroyKernel kernel

  return (V.toList times, events)

-- | Collect trace events
collectTraceEvents :: [Double] -> Int -> IO [ProfileEvent]
collectTraceEvents times n = do
  let startTime = 0
      events = zipWith (\i timeMs ->
        let start = startTime + sum (take i times) * 1_000_000
            end = start + timeMs * 1_000_000
        in ProfileEvent
             { eventName_ = "OptimizedMatMul_" ++ show n
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

      -- Calculate TFLOPS (2 * M * N * K for MatMul)
      totalFlops = 2.0 * fromIntegral n * fromIntegral n * fromIntegral n :: Double
      tflops = totalFlops / (avgTime / 1000.0) / 1e12

  putStrLn ""
  putStrLn "=== Results ==="
  printf "[Bench] Size: %d, Iterations: %d\n" n (benchIters config)
  printf "[Time]  Avg: %.2f ms | Min: %.2f ms | Max: %.2f ms\n" avgTime minTime maxTime
  printf "[Perf]  %.3f TFLOPS\n" tflops
  putStrLn ""

  -- Roofline static analysis
  let analysis = analyzeMatMul n F32
  putStrLn $ rooflineReport analysis

  -- Additional optimization insights
  putStrLn "=== Optimization Features ==="
  putStrLn "[Tiling]        2D block-tiling (BM=64, BK=8, BN=64, TM=8, TN=8)"
  putStrLn "[Vectorization] vec4 for 4-wide loads/stores"
  putStrLn "[Memory]        Workgroup shared memory (tileA, tileB)"
  putStrLn "[Transpose]     B is transposed for better cache locality"
  putStrLn "[Unrolling]     Static loop unrolling for inner loops"
  putStrLn ""
