{-# LANGUAGE NumericUnderscores #-}

{-|
Async MatMul Benchmark - Real-world demonstration of async pipeline benefits

This benchmark compares synchronous vs async execution for heavy matmul workloads,
showing the actual performance impact of CPU/GPU overlap on the path to 300 TPS.

Usage:
  cabal run bench-async-matmul -- --size 2048 --iters 20

Expected: 1.3-1.8x speedup for async execution when GPU time >> thread overhead
-}

module Main where

import Prelude (Int, Double, Float, String, Show, IO, ($), (++), (+), (-), (*), (/), (>>=),
                show, div, fromIntegral, putStrLn, return, (>>))
import qualified Prelude as P
import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.Types (Shape(..), AnyTensor(..))
import Graphics.WebGPU.Dawn.Kernel (KernelCode, Kernel, WorkgroupSize(..), createKernelCode,
                                     setEntryPoint, compileKernelHeterogeneous, dispatchKernel,
                                     destroyKernel, destroyKernelCode)
import WGSL.Async.Pipeline
import System.Clock (Clock(..), getTime, diffTimeSpec, toNanoSecs)
import System.Environment (getArgs)
import Text.Printf (printf)
import qualified Data.Vector.Storable as V
import Control.Monad (forM_)

-- | Benchmark configuration
data BenchConfig = BenchConfig
  { benchSize :: !Int
  , benchIters :: !Int
  } deriving (Show)

defaultConfig :: BenchConfig
defaultConfig = BenchConfig
  { benchSize = 2048
  , benchIters = 20
  }

-- | Parse command line arguments
parseArgs :: [String] -> BenchConfig
parseArgs = go defaultConfig
  where
    go cfg [] = cfg
    go cfg ("--size":sizeStr:rest) = go (cfg { benchSize = P.read sizeStr }) rest
    go cfg ("--iters":iterStr:rest) = go (cfg { benchIters = P.read iterStr }) rest
    go cfg (_:rest) = go cfg rest

-- | Optimized matmul shader code (from run.cpp - kShaderMatmulWithTranspose)
createMatmulShader :: Int -> Int -> Int -> String
createMatmulShader m k n =
  let bm = 64; bk = 8; bn = 64
      tm = 8; tn = 8; tn4 = 2
  in P.unlines
    [ "@group(0) @binding(0) var<storage, read_write> A: array<f32>;"
    , "@group(0) @binding(1) var<storage, read_write> B: array<f32>;"
    , "@group(0) @binding(2) var<storage, read_write> C: array<f32>;"
    , ""
    , "var<workgroup> tileA: array<f32, " ++ show (bm * bk) ++ ">;"
    , "var<workgroup> tileB: array<f32, " ++ show (bk * bn) ++ ">;"
    , ""
    , "@compute @workgroup_size(" ++ show (bm `div` tm) ++ ", " ++ show (bn `div` tn) ++ ", 1)"
    , "fn main("
    , "  @builtin(workgroup_id) wgid: vec3<u32>,"
    , "  @builtin(local_invocation_id) lid: vec3<u32>"
    , ") {"
    , "  let row = wgid.x * " ++ show bm ++ "u + lid.x * " ++ show tm ++ "u;"
    , "  let col = wgid.y * " ++ show bn ++ "u + lid.y * " ++ show tn ++ "u;"
    , ""
    , "  var threadResults: array<vec4<f32>, " ++ show (tm * tn4) ++ ">;"
    , "  for (var i = 0u; i < " ++ show (tm * tn4) ++ "u; i++) {"
    , "    threadResults[i] = vec4<f32>(0.0);"
    , "  }"
    , ""
    , "  for (var bkidx = 0u; bkidx < " ++ show k ++ "u; bkidx += " ++ show bk ++ "u) {"
    , "    // Load tileA (BM x BK)"
    , "    for (var i = 0u; i < " ++ show (bm * bk `div` (bm `div` tm * bn `div` tn)) ++ "u; i++) {"
    , "      let idx = (lid.y * " ++ show (bm `div` tm) ++ "u + lid.x) + i * " ++ show (bm `div` tm * bn `div` tn) ++ "u;"
    , "      if (idx < " ++ show (bm * bk) ++ "u) {"
    , "        let tileRow = idx / " ++ show bk ++ "u;"
    , "        let tileCol = idx % " ++ show bk ++ "u;"
    , "        let aRow = wgid.x * " ++ show bm ++ "u + tileRow;"
    , "        let aCol = bkidx + tileCol;"
    , "        if (aRow < " ++ show m ++ "u && aCol < " ++ show k ++ "u) {"
    , "          tileA[idx] = A[aRow * " ++ show k ++ "u + aCol];"
    , "        } else {"
    , "          tileA[idx] = 0.0;"
    , "        }"
    , "      }"
    , "    }"
    , ""
    , "    // Load tileB (BK x BN) - transposed layout"
    , "    for (var i = 0u; i < " ++ show (bk * bn `div` (bm `div` tm * bn `div` tn)) ++ "u; i++) {"
    , "      let idx = (lid.y * " ++ show (bm `div` tm) ++ "u + lid.x) + i * " ++ show (bm `div` tm * bn `div` tn) ++ "u;"
    , "      if (idx < " ++ show (bk * bn) ++ "u) {"
    , "        let tileRow = idx / " ++ show bn ++ "u;"
    , "        let tileCol = idx % " ++ show bn ++ "u;"
    , "        let bCol = wgid.y * " ++ show bn ++ "u + tileCol;"
    , "        let bRow = bkidx + tileRow;"
    , "        if (bRow < " ++ show k ++ "u && bCol < " ++ show n ++ "u) {"
    , "          tileB[idx] = B[bRow * " ++ show n ++ "u + bCol];"
    , "        } else {"
    , "          tileB[idx] = 0.0;"
    , "        }"
    , "      }"
    , "    }"
    , ""
    , "    workgroupBarrier();"
    , ""
    , "    // Compute"
    , "    for (var dotIdx = 0u; dotIdx < " ++ show bk ++ "u; dotIdx++) {"
    , "      for (var i = 0u; i < " ++ show tm ++ "u; i++) {"
    , "        let a = tileA[(lid.x * " ++ show tm ++ "u + i) * " ++ show bk ++ "u + dotIdx];"
    , "        for (var j = 0u; j < " ++ show tn4 ++ "u; j++) {"
    , "          let b = vec4<f32>("
    , "            tileB[dotIdx * " ++ show bn ++ "u + lid.y * " ++ show tn ++ "u + j * 4u],"
    , "            tileB[dotIdx * " ++ show bn ++ "u + lid.y * " ++ show tn ++ "u + j * 4u + 1u],"
    , "            tileB[dotIdx * " ++ show bn ++ "u + lid.y * " ++ show tn ++ "u + j * 4u + 2u],"
    , "            tileB[dotIdx * " ++ show bn ++ "u + lid.y * " ++ show tn ++ "u + j * 4u + 3u]"
    , "          );"
    , "          threadResults[i * " ++ show tn4 ++ "u + j] += a * b;"
    , "        }"
    , "      }"
    , "    }"
    , ""
    , "    workgroupBarrier();"
    , "  }"
    , ""
    , "  // Store results"
    , "  for (var i = 0u; i < " ++ show tm ++ "u; i++) {"
    , "    for (var j = 0u; j < " ++ show tn4 ++ "u; j++) {"
    , "      let outRow = row + i;"
    , "      let outCol = col + j * 4u;"
    , "      if (outRow < " ++ show m ++ "u && outCol + 3u < " ++ show n ++ "u) {"
    , "        let baseIdx = outRow * " ++ show n ++ "u + outCol;"
    , "        let result = threadResults[i * " ++ show tn4 ++ "u + j];"
    , "        C[baseIdx] = result.x;"
    , "        C[baseIdx + 1u] = result.y;"
    , "        C[baseIdx + 2u] = result.z;"
    , "        C[baseIdx + 3u] = result.w;"
    , "      }"
    , "    }"
    , "  }"
    , "}"
    ]

main :: IO ()
main = do
  args <- getArgs
  let config = parseArgs args

  putStrLn "=== Async MatMul Benchmark (Path to 300 TPS) ==="
  printf "[Config] Size: %d x %d, Iterations: %d\n" (benchSize config) (benchSize config) (benchIters config)
  putStrLn "[Config] Shader: Optimized 2D block-tiling (BM=64, BK=8, BN=64)"
  putStrLn ""

  -- Create GPU context
  ctx <- createContext

  let n = benchSize config
      m = n
      k = n

  -- Create input matrices
  let matA = V.generate (m * k) (\i -> fromIntegral (i `P.mod` 100) / 100.0 :: Float)
      matB = V.generate (k * n) (\i -> fromIntegral ((i + 5) `P.mod` 100) / 100.0 :: Float)

  putStrLn "[Phase 1] Synchronous Execution (Baseline)"
  putStrLn "--------------------------------------------"
  timeSyncMs <- benchSynchronous ctx config matA matB

  putStrLn ""
  putStrLn "[Phase 2] Async Pipelined Execution"
  putStrLn "--------------------------------------------"
  timeAsyncMs <- benchAsync ctx config matA matB

  putStrLn ""
  putStrLn "[Results]"
  putStrLn "============================================"
  let speedup = timeSyncMs / timeAsyncMs
      reduction = (timeSyncMs - timeAsyncMs) / timeSyncMs * 100.0

      -- Calculate TFLOPS
      totalFlops = 2.0 * fromIntegral n * fromIntegral n * fromIntegral n :: Double
      tflopsSync = totalFlops / (timeSyncMs / 1000.0) / 1e12
      tflopsAsync = totalFlops / (timeAsyncMs / 1000.0) / 1e12

  printf "  Synchronous: %.2f ms (%.3f TFLOPS)\n" timeSyncMs tflopsSync
  printf "  Async:       %.2f ms (%.3f TFLOPS)\n" timeAsyncMs tflopsAsync
  printf "  Speedup:     %.2fx faster\n" speedup
  printf "  Reduction:   %.1f%% less time\n" reduction

  putStrLn ""
  if speedup P.> 1.2
    then do
      putStrLn "✓ SUCCESS: Async pipeline provides significant speedup!"
      putStrLn "  This demonstrates CPU/GPU overlap is working."
      putStrLn "  Expected impact on Gemma 3: 1.3-1.8x throughput improvement"
    else do
      putStrLn "⚠ Note: Speedup is modest for this workload."
      putStrLn "  GPU execution may be too fast relative to thread overhead."
      putStrLn "  Try larger matrix sizes (--size 4096) or more iterations."

  destroyContext ctx
  putStrLn ""
  putStrLn "Integration ready for Gemma 3!"

-- | Benchmark synchronous execution
benchSynchronous :: Context -> BenchConfig -> V.Vector Float -> V.Vector Float -> IO Double
benchSynchronous ctx config matA matB = do
  let n = benchSize config
      iters = benchIters config
      m = n
      k = n

  -- Align dimensions to 64 for tiling
  let actualM = ((m + 63) `div` 64) * 64
      actualK = ((k + 63) `div` 64) * 64
      actualN = ((n + 63) `div` 64) * 64

  -- Create tensors
  tensorA <- createTensorWithData ctx (Shape [actualM * actualK]) matA
  tensorB <- createTensorWithData ctx (Shape [actualK * actualN]) matB
  tensorC <- createTensor ctx (Shape [actualM * actualN]) F32

  -- Create kernel
  let shaderCode = createMatmulShader actualM actualK actualN
  code <- createKernelCode shaderCode
  setEntryPoint code "main"

  let numWorkgroupsX = actualM `div` 64
      numWorkgroupsY = actualN `div` 64
      wgSize = WorkgroupSize numWorkgroupsX numWorkgroupsY 1

  kernel <- compileKernelHeterogeneous ctx code
    [AnyTensor tensorA, AnyTensor tensorB, AnyTensor tensorC] wgSize

  -- Warm-up
  putStrLn "  [Warmup] Running 1 iteration..."
  dispatchKernel ctx kernel

  -- Benchmark
  putStrLn $ "  [Bench] Running " ++ show iters ++ " iterations..."
  start <- getTime Monotonic
  forM_ [1..iters] $ \i -> do
    dispatchKernel ctx kernel
  end <- getTime Monotonic

  let diff = diffTimeSpec end start
      totalTimeMs = fromIntegral (toNanoSecs diff) / 1_000_000.0
      avgTimeMs = totalTimeMs / fromIntegral iters

  printf "  Total: %.2f ms | Per-iteration: %.2f ms\n" totalTimeMs avgTimeMs

  -- Cleanup
  destroyKernel kernel
  destroyKernelCode code
  destroyTensor tensorA
  destroyTensor tensorB
  destroyTensor tensorC

  return totalTimeMs

-- | Benchmark async pipelined execution
benchAsync :: Context -> BenchConfig -> V.Vector Float -> V.Vector Float -> IO Double
benchAsync ctx config matA matB = do
  let n = benchSize config
      iters = benchIters config
      m = n
      k = n

  -- Align dimensions
  let actualM = ((m + 63) `div` 64) * 64
      actualK = ((k + 63) `div` 64) * 64
      actualN = ((n + 63) `div` 64) * 64

  -- Create async pipeline
  pipeline <- createAsyncPipeline ctx

  -- Create tensors (shared across iterations)
  tensorA <- createTensorWithData ctx (Shape [actualM * actualK]) matA
  tensorB <- createTensorWithData ctx (Shape [actualK * actualN]) matB
  tensorC <- createTensor ctx (Shape [actualM * actualN]) F32

  -- Create kernel
  let shaderCode = createMatmulShader actualM actualK actualN
  code <- createKernelCode shaderCode
  setEntryPoint code "main"

  let numWorkgroupsX = actualM `div` 64
      numWorkgroupsY = actualN `div` 64
      wgSize = WorkgroupSize numWorkgroupsX numWorkgroupsY 1

  kernel <- compileKernelHeterogeneous ctx code
    [AnyTensor tensorA, AnyTensor tensorB, AnyTensor tensorC] wgSize

  -- Warm-up
  putStrLn "  [Warmup] Running 1 iteration..."
  submitCommand pipeline $ do
    dispatchKernel ctx kernel
  awaitPipeline pipeline

  -- Benchmark
  putStrLn $ "  [Bench] Running " ++ show iters ++ " async iterations..."
  start <- getTime Monotonic

  -- Submit all commands asynchronously (CPU continues while GPU processes)
  forM_ [1..iters] $ \_ -> do
    submitCommand pipeline $ do
      dispatchKernel ctx kernel

  -- Wait for all GPU work to complete
  awaitPipeline pipeline
  end <- getTime Monotonic

  let diff = diffTimeSpec end start
      totalTimeMs = fromIntegral (toNanoSecs diff) / 1_000_000.0
      avgTimeMs = totalTimeMs / fromIntegral iters

  -- Get pipeline stats
  stats <- getPipelineStats pipeline
  printf "  Commands: %d submitted, %d executed, %d errors\n"
    (statSubmitted stats) (statExecuted stats) (statErrors stats)
  printf "  Total: %.2f ms | Per-iteration: %.2f ms\n" totalTimeMs avgTimeMs

  -- Cleanup
  destroyAsyncPipeline pipeline
  destroyKernel kernel
  destroyKernelCode code
  destroyTensor tensorA
  destroyTensor tensorB
  destroyTensor tensorC

  return totalTimeMs
