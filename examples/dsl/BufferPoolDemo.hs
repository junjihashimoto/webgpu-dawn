{-# LANGUAGE NumericUnderscores #-}

{-|
Buffer Pool Demo - Demonstrating GPU Buffer Recycling

This demo shows how the buffer pool reduces allocation overhead by recycling
GPU buffers instead of creating/destroying them repeatedly.

Usage:
  cabal run buffer-pool-demo

Expected output:
- Statistics showing cache hit rate (should be ~90% after warmup)
- Performance comparison: pooled vs non-pooled allocations
-}

module Main where

import Prelude (Int, Double, IO, ($), (++), (+), (*), (/), show, fromIntegral,
                putStrLn, return, (>>=), (>>))
import qualified Prelude as P
import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.Types (BufferUsage(..))
import WGSL.Memory.Pool
import System.Clock (Clock(..), getTime, diffTimeSpec, toNanoSecs)
import Text.Printf (printf)

-- | Number of buffer allocations to test
numAllocations :: Int
numAllocations = 100

-- | Buffer sizes to test (in bytes)
testSizes :: [Int]
testSizes = [1024, 4096, 16384, 65536]

-- | Benchmark buffer pool performance
main :: IO ()
main = do
  putStrLn "=== Buffer Pool Demo ==="
  putStrLn ""

  -- Create GPU context
  ctx <- createContext

  -- Create buffer pool
  pool <- createBufferPool

  putStrLn "[Phase 1] Testing buffer pool with recycling"
  putStrLn "--------------------------------------------"

  -- Reset stats
  resetPoolStats pool

  -- Acquire and release buffers (simulates workload)
  startPooled <- getTime Monotonic
  testWithPool ctx pool
  endPooled <- getTime Monotonic

  -- Get pool statistics
  stats <- getPoolStats pool
  let hitRate = if statAcquired stats P.> 0
                then fromIntegral (statReused stats) * 100.0 / fromIntegral (statAcquired stats) :: Double
                else 0.0

  putStrLn ""
  putStrLn "[Pool Statistics]"
  printf "  Acquired:  %d buffers\n" (statAcquired stats)
  printf "  Reused:    %d buffers (%.1f%% hit rate)\n" (statReused stats) hitRate
  printf "  Created:   %d buffers\n" (statCreated stats)
  printf "  Released:  %d buffers\n" (statReleased stats)

  let diffPooled = diffTimeSpec endPooled startPooled
      timePooledMs = fromIntegral (toNanoSecs diffPooled) / 1_000_000.0 :: Double
  printf "  Time:      %.2f ms\n" timePooledMs

  putStrLn ""
  putStrLn "[Phase 2] Testing without pool (baseline)"
  putStrLn "--------------------------------------------"

  -- Benchmark without pool
  startNonPooled <- getTime Monotonic
  testWithoutPool ctx
  endNonPooled <- getTime Monotonic

  let diffNonPooled = diffTimeSpec endNonPooled startNonPooled
      timeNonPooledMs = fromIntegral (toNanoSecs diffNonPooled) / 1_000_000.0 :: Double
  printf "  Time:      %.2f ms\n" timeNonPooledMs

  putStrLn ""
  putStrLn "[Comparison]"
  putStrLn "--------------------------------------------"
  let speedup = timeNonPooledMs / timePooledMs
      reduction = (timeNonPooledMs - timePooledMs) / timeNonPooledMs * 100.0
  printf "  Speedup:   %.2fx faster with pool\n" speedup
  printf "  Reduction: %.1f%% less overhead\n" reduction

  putStrLn ""
  if hitRate P.> 80.0
    then putStrLn "✓ Pool working correctly! High cache hit rate achieved."
    else putStrLn "⚠ Warning: Low cache hit rate. Pool may not be working as expected."

  -- Cleanup
  destroyBufferPool pool
  destroyContext ctx

  putStrLn ""
  putStrLn "Demo complete!"

-- | Test with buffer pool (acquires, releases, reuses)
testWithPool :: Context -> BufferPool -> IO ()
testWithPool ctx pool = do
  -- Run multiple iterations to show recycling benefit
  P.mapM_ (\size -> do
    -- Acquire buffers
    buffers <- P.mapM (\_ -> acquireBuffer pool ctx (fromIntegral size) StorageBuffer) [1..numAllocations]

    -- Release buffers back to pool
    P.mapM_ (releaseBuffer pool) buffers

    -- Acquire again (should hit cache!)
    buffers2 <- P.mapM (\_ -> acquireBuffer pool ctx (fromIntegral size) StorageBuffer) [1..numAllocations]

    -- Release again
    P.mapM_ (releaseBuffer pool) buffers2

    return ()
    ) testSizes

-- | Test without pool (create/destroy every time)
testWithoutPool :: Context -> IO ()
testWithoutPool ctx = do
  P.mapM_ (\size -> do
    -- Create buffers
    buffers <- P.mapM (\_ -> createBuffer ctx (fromIntegral size) StorageBuffer) [1..numAllocations]

    -- Destroy buffers
    P.mapM_ destroyBuffer buffers

    -- Create again (no recycling!)
    buffers2 <- P.mapM (\_ -> createBuffer ctx (fromIntegral size) StorageBuffer) [1..numAllocations]

    -- Destroy again
    P.mapM_ destroyBuffer buffers2

    return ()
    ) testSizes
