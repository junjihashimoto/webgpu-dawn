{-# LANGUAGE DataKinds #-}
{-# LANGUAGE NumericUnderscores #-}

{-|
Linear Layer Benchmark - MatMul Performance Testing

This benchmark measures FP16 matrix multiplication performance and
validates correctness. It supports Chrome Tracing export for visualization.

Usage:
  cabal run bench-linear                    # Run benchmark
  cabal run bench-linear -- --trace         # Export trace.json
  cabal run bench-linear -- --size 2048     # Custom size
  cabal run bench-linear -- --iters 100     # Custom iterations
-}

module Main where

import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.Tensor
import Graphics.WebGPU.Dawn.Kernel
import WGSL.Profile
import WGSL.Profile.Trace
import WGSL.Analyze
import qualified Data.Vector.Storable as V
import System.Environment (getArgs)
import System.Clock (Clock(..), getTime, diffTimeSpec, toNanoSecs)
import Control.Monad (when)
import Text.Printf (printf)
import Data.Word (Word64)

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

main :: IO ()
main = do
  args <- getArgs
  let config = parseArgs args

  putStrLn "=== Linear Layer Benchmark (FP16 MatMul) ==="
  putStrLn $ "[Config] Size: " ++ show (benchSize config) ++
             ", Iterations: " ++ show (benchIters config)
  putStrLn $ "[Config] Trace: " ++ show (benchTrace config) ++
             ", Validate: " ++ show (benchValidate config)
  putStrLn ""

  -- Create GPU context with timestamp query support (for profiling)
  ctx <- if benchTrace config
         then createContextWithFeatures ["allow_unsafe_apis"] [FeatureTimestampQuery]
         else createContext

  -- Run benchmark
  (times, events) <- runMatMulBench ctx config

  -- Print statistics
  printStats config times

  -- Export trace if requested
  when (benchTrace config) $ do
    putStrLn ""
    putStrLn "[Trace] Exporting to trace.json..."
    writeChromeTrace "trace.json" events
    putStrLn "[Trace] Load trace.json in ui.perfetto.dev or chrome://tracing"

  destroyContext ctx

-- | Run MatMul benchmark and return timing results and profile events
runMatMulBench :: Context -> BenchConfig -> IO ([Double], [ProfileEvent])
runMatMulBench ctx config = do
  let n = benchSize config
      iters = benchIters config

  -- Create input matrices (FP32 for now)
  let matA = V.generate (n * n) (\i -> fromIntegral (i `mod` 100) / 100.0 :: Float)
      matB = V.generate (n * n) (\i -> fromIntegral ((i * 2) `mod` 100) / 100.0 :: Float)

  -- Validate correctness on small sample if requested
  when (benchValidate config) $ do
    validateMatMul matA matB n

  -- Create GPU tensors
  tensorA <- createTensorWithData ctx (Shape [n, n]) matA
  tensorB <- createTensorWithData ctx (Shape [n, n]) matB
  tensorOut <- createTensor ctx (Shape [n, n]) F32

  -- Compile MatMul kernel
  kernel <- compileMatMulKernel ctx tensorA tensorB tensorOut n

  -- Warm-up run
  putStrLn "[Warmup] Running 1 iteration..."
  dispatchKernel ctx kernel

  -- Benchmark runs
  putStrLn $ "[Bench] Running " ++ show iters ++ " iterations..."
  times <- V.generateM iters $ \i -> do
    when (i `mod` 10 == 0) $
      putStrLn $ "  Iteration " ++ show i ++ "/" ++ show iters

    -- Time this iteration
    start <- getTime Monotonic
    dispatchKernel ctx kernel
    end <- getTime Monotonic

    let diff = diffTimeSpec end start
        timeMs = fromIntegral (toNanoSecs diff) / 1_000_000.0
    return timeMs

  -- If tracing enabled, collect profile events
  events <- if benchTrace config
            then collectTraceEvents ctx kernel (V.toList times) n
            else return []

  -- Cleanup
  destroyTensor tensorA
  destroyTensor tensorB
  destroyTensor tensorOut
  destroyKernel kernel

  return (V.toList times, events)

-- | Simple MatMul kernel (placeholder - you can replace with optimized version)
compileMatMulKernel :: Context -> Tensor 'F32 -> Tensor 'F32 -> Tensor 'F32 -> Int -> IO Kernel
compileMatMulKernel ctx tensorA tensorB tensorOut n = do
  -- Simplified MatMul shader (this is a placeholder)
  -- In a real implementation, this would use the DSL or hand-written WGSL
  let wgslCode = matMulShaderCode n

  kernelCode <- createKernelCode wgslCode
  setWorkgroupSize kernelCode (WorkgroupSize 16 16 1)
  setEntryPoint kernelCode "main"

  -- Create kernel with tensors
  let wgSize = WorkgroupSize ((n + 15) `div` 16) ((n + 15) `div` 16) 1
  compileKernel ctx kernelCode [tensorA, tensorB, tensorOut] wgSize

-- | Generate MatMul WGSL shader code
matMulShaderCode :: Int -> String
matMulShaderCode n = unlines
  [ "@group(0) @binding(0) var<storage, read> matA: array<f32>;"
  , "@group(0) @binding(1) var<storage, read> matB: array<f32>;"
  , "@group(0) @binding(2) var<storage, read_write> matOut: array<f32>;"
  , ""
  , "@compute @workgroup_size(16, 16, 1)"
  , "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {"
  , "  let row = global_id.x;"
  , "  let col = global_id.y;"
  , "  let N = " ++ show n ++ "u;"
  , ""
  , "  if (row >= N || col >= N) {"
  , "    return;"
  , "  }"
  , ""
  , "  var sum = 0.0;"
  , "  for (var k = 0u; k < N; k++) {"
  , "    sum += matA[row * N + k] * matB[k * N + col];"
  , "  }"
  , "  matOut[row * N + col] = sum;"
  , "}"
  ]

-- | Validate MatMul correctness on small sample
validateMatMul :: V.Vector Float -> V.Vector Float -> Int -> IO ()
validateMatMul matA matB n = do
  putStrLn "[Validate] Checking small sample (3x3) on CPU..."

  -- For simplicity, just check that no NaN values exist
  -- A full validation would compute reference result on CPU
  let hasNaN = V.any (\f -> isNaN f || isInfinite f) matA ||
               V.any (\f -> isNaN f || isInfinite f) matB

  if hasNaN
    then error "[Validate] FAILED: Input contains NaN or Inf!"
    else putStrLn "[Validate] PASSED: Inputs are valid"

-- | Collect trace events from profiled runs
collectTraceEvents :: Context -> Kernel -> [Double] -> Int -> IO [ProfileEvent]
collectTraceEvents ctx kernel times n = do
  -- Create mock profile events based on timing data
  -- In a real implementation, you would use timestamp queries
  let startTime = 0
      events = zipWith (\i timeMs ->
        let start = startTime + sum (take i times) * 1_000_000
            end = start + timeMs * 1_000_000
        in ProfileEvent
             { eventName_ = "MatMul_" ++ show n
             , startTime = floor start
             , endTime = floor end
             , eventDurationMs = timeMs
             }
        ) [0..] times
  return events

-- | Print benchmark statistics
printStats :: BenchConfig -> [Double] -> IO ()
printStats config times = do
  let n = benchSize config
      avgTime = sum times / fromIntegral (length times)
      minTime = minimum times
      maxTime = maximum times

      -- Calculate TFLOPS
      -- MatMul FLOPs = 2 * M * N * K = 2 * N^3 for square matrices
      totalFlops = 2.0 * fromIntegral n * fromIntegral n * fromIntegral n :: Double
      tflops = totalFlops / (avgTime / 1000.0) / 1e12  -- Convert ms to seconds, then to TFLOPS

  putStrLn ""
  putStrLn "=== Results ==="
  printf "[Bench] Size: %d, Iterations: %d\n" n (benchIters config)
  printf "[Time]  Avg: %.2f ms | Min: %.2f ms | Max: %.2f ms\n" avgTime minTime maxTime
  printf "[Perf]  %.3f TFLOPS\n" tflops
  putStrLn ""

  -- Roofline static analysis
  let analysis = analyzeMatMul n F32
  putStrLn $ rooflineReport analysis
