{-# LANGUAGE NumericUnderscores #-}
{-# LANGUAGE DataKinds #-}

{-|
Async Pipeline Demo - Demonstrating CPU/GPU Overlap

This demo shows how the async pipeline enables latency hiding by overlapping
CPU encoding with GPU execution.

Usage:
  cabal run async-pipeline-demo

Expected output:
- Performance comparison: sequential vs pipelined execution
- Speedup should be ~1.5-2x for pipelined version
-}

module Main where

import Prelude (Int, Double, Float, IO, ($), (++), (+), (-), (*), (/), (>>=), show, fromIntegral,
                putStrLn, return, (>>))
import qualified Prelude as P
import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.Types (Shape(..), AnyTensor(..))
import WGSL.DSL
import WGSL.Execute (executeShaderNamed)
import WGSL.Async.Pipeline
import System.Clock (Clock(..), getTime, diffTimeSpec, toNanoSecs)
import Text.Printf (printf)
import qualified Data.Vector.Storable as V
import Control.Monad (forM_)

-- | Number of iterations for benchmark
numIterations :: Int
numIterations = 20

-- | Vector size for testing
vectorSize :: Int
vectorSize = 1_000_000

-- | Simple vector addition shader for testing
-- Note: This is a simplified shader just for testing async pipeline
-- The actual computation is placeholder - focus is on pipeline performance
vectorAddShader :: ShaderM ()
vectorAddShader = do
  vecA <- declareInputBuffer "A" (TArray vectorSize TF32)
  vecB <- declareInputBuffer "B" (TArray vectorSize TF32)
  vecC <- declareOutputBuffer "C" (TArray vectorSize TF32)

  gid <- globalId
  let idx = vecX gid

  -- Placeholder: Just create a simple computation
  -- This tests the pipeline overhead, not shader complexity
  result <- var TF32 (litF32 1.0)
  assign result ((litF32 2.0) + (litF32 3.0))

  return ()

main :: IO ()
main = do
  putStrLn "=== Async Pipeline Demo ==="
  putStrLn ""
  putStrLn "[Config] Comparing sequential vs pipelined execution"
  printf "[Config] Iterations: %d, Vector size: %d\n" numIterations vectorSize
  putStrLn ""

  -- Create GPU context
  ctx <- createContext

  -- Create test data
  let vecA = V.generate vectorSize (\i -> fromIntegral i :: Float)
      vecB = V.generate vectorSize (\i -> fromIntegral (i + 1) :: Float)

  -- Create input tensors (shared across both tests)
  tensorA <- createTensorWithData ctx (Shape [vectorSize]) vecA
  tensorB <- createTensorWithData ctx (Shape [vectorSize]) vecB

  -- Build shader
  let shaderModule = buildShaderWithAutoBinding (256, 1, 1) vectorAddShader
      workgroupSize = WorkgroupSize ((vectorSize + 255) `P.div` 256) 1 1

  putStrLn "[Phase 1] Sequential execution (baseline)"
  putStrLn "--------------------------------------------"
  timeSeq <- benchmarkSequential ctx shaderModule tensorA tensorB workgroupSize

  putStrLn ""
  putStrLn "[Phase 2] Pipelined execution (async)"
  putStrLn "--------------------------------------------"
  timeAsync <- benchmarkAsync ctx shaderModule tensorA tensorB workgroupSize

  putStrLn ""
  putStrLn "[Comparison]"
  putStrLn "--------------------------------------------"
  let speedup = timeSeq / timeAsync
      reduction = (timeSeq - timeAsync) / timeSeq * 100.0
  printf "  Sequential: %.2f ms\n" timeSeq
  printf "  Pipelined:  %.2f ms\n" timeAsync
  printf "  Speedup:    %.2fx faster\n" speedup
  printf "  Reduction:  %.1f%% less time\n" reduction

  putStrLn ""
  if speedup P.> 1.2
    then putStrLn "✓ Pipeline working! Significant speedup achieved."
    else putStrLn "⚠ Warning: Low speedup. Pipeline may not be overlapping effectively."

  -- Cleanup
  destroyTensor tensorA
  destroyTensor tensorB
  destroyContext ctx

  putStrLn ""
  putStrLn "Demo complete!"

-- | Benchmark sequential execution
benchmarkSequential :: Context -> ShaderModule -> Tensor 'F32 -> Tensor 'F32 -> WorkgroupSize -> IO Double
benchmarkSequential ctx shader tensorA tensorB wgSize = do
  -- Warm-up
  tensorC <- createTensor ctx (Shape [vectorSize]) F32
  executeShaderNamed ctx shader
    [ ("A", AnyTensor tensorA)
    , ("B", AnyTensor tensorB)
    , ("C", AnyTensor tensorC)
    ] wgSize
  destroyTensor tensorC

  -- Benchmark
  start <- getTime Monotonic
  forM_ [1..numIterations] $ \_ -> do
    tensorC <- createTensor ctx (Shape [vectorSize]) F32
    executeShaderNamed ctx shader
      [ ("A", AnyTensor tensorA)
      , ("B", AnyTensor tensorB)
      , ("C", AnyTensor tensorC)
      ] wgSize
    destroyTensor tensorC
  end <- getTime Monotonic

  let diff = diffTimeSpec end start
      timeMs = fromIntegral (toNanoSecs diff) / 1_000_000.0
  printf "  Total time: %.2f ms\n" timeMs
  printf "  Per iteration: %.2f ms\n" (timeMs / fromIntegral numIterations)
  return timeMs

-- | Benchmark async pipelined execution
benchmarkAsync :: Context -> ShaderModule -> Tensor 'F32 -> Tensor 'F32 -> WorkgroupSize -> IO Double
benchmarkAsync ctx shader tensorA tensorB wgSize = do
  -- Create async pipeline
  pipeline <- createAsyncPipeline ctx

  -- Warm-up
  submitCommand pipeline $ do
    tensorC <- createTensor ctx (Shape [vectorSize]) F32
    executeShaderNamed ctx shader
      [ ("A", AnyTensor tensorA)
      , ("B", AnyTensor tensorB)
      , ("C", AnyTensor tensorC)
      ] wgSize
    destroyTensor tensorC
  awaitPipeline pipeline

  -- Benchmark
  start <- getTime Monotonic
  forM_ [1..numIterations] $ \_ -> do
    -- Submit command (non-blocking) - CPU continues immediately
    submitCommand pipeline $ do
      tensorC <- createTensor ctx (Shape [vectorSize]) F32
      executeShaderNamed ctx shader
        [ ("A", AnyTensor tensorA)
        , ("B", AnyTensor tensorB)
        , ("C", AnyTensor tensorC)
        ] wgSize
      destroyTensor tensorC

  -- Wait for all commands to complete
  awaitPipeline pipeline
  end <- getTime Monotonic

  -- Get stats
  stats <- getPipelineStats pipeline
  printf "  Commands submitted: %d\n" (statSubmitted stats)
  printf "  Commands executed:  %d\n" (statExecuted stats)
  printf "  Errors:             %d\n" (statErrors stats)

  let diff = diffTimeSpec end start
      timeMs = fromIntegral (toNanoSecs diff) / 1_000_000.0
  printf "  Total time: %.2f ms\n" timeMs
  printf "  Per iteration: %.2f ms\n" (timeMs / fromIntegral numIterations)

  -- Cleanup pipeline
  destroyAsyncPipeline pipeline

  return timeMs
