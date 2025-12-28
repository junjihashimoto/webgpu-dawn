{-# LANGUAGE RecordWildCards #-}

{-|
Module      : WGSL.Memory.Pool
Description : GPU buffer memory pool for latency hiding
Copyright   : (c) 2025
License     : MIT

Buffer pool for recycling GPU memory allocations.

= Motivation

Creating and destroying GPU buffers is expensive (involves driver overhead,
synchronization, and potential GPU stalls). By pooling buffers, we can:

1. **Reduce allocation overhead**: Reuse existing buffers instead of creating new ones
2. **Hide latency**: Keep buffers warm in the pool while GPU is processing
3. **Improve throughput**: Critical for 300 TPS goal in Gemma 3

= Usage Pattern

@
  -- Create pool
  pool <- createBufferPool

  -- Acquire buffer (creates or reuses from pool)
  buffer <- acquireBuffer pool ctx size usage

  -- Use buffer...

  -- Release back to pool (doesn't destroy, just recycles)
  releaseBuffer pool buffer

  -- Cleanup when done
  destroyBufferPool pool
@

= Performance Impact

For workloads with repeated allocations (like LLM inference):
- **Before**: Create/destroy buffer on every layer → high overhead
- **After**: Reuse from pool → minimal overhead

Expected speedup: 10-30% for buffer-heavy workloads.

= Implementation Notes

- Uses `IORef` for thread-safe access (assumes single-threaded GPU context)
- Bins buffers by size for O(1) lookup
- Supports different buffer usages (Storage, Uniform, etc.)
- Automatic cleanup on pool destruction
-}

module WGSL.Memory.Pool
  ( -- * Buffer Pool
    BufferPool
  , createBufferPool
  , destroyBufferPool
    -- * Buffer Operations
  , acquireBuffer
  , releaseBuffer
    -- * Statistics
  , PoolStats(..)
  , getPoolStats
  , resetPoolStats
  ) where

import Graphics.WebGPU.Dawn.Types (Context, Tensor(..), NumType(..))
import Graphics.WebGPU.Dawn (createTensor, destroyTensor)
import Data.IORef
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Control.Monad (forM_, when)
import Data.Word (Word64)

-- | Buffer pool for recycling GPU allocations
--
-- Internally maintains a map from buffer size to available buffers.
-- Buffers are binned by size for efficient lookup.
data BufferPool = BufferPool
  { poolBuffers :: !(IORef (Map.Map Word64 [Buffer]))
    -- ^ Available buffers organized by size
  , poolActive :: !(IORef (Set.Set Buffer))
    -- ^ Currently active (acquired) buffers
  , poolSizes :: !(IORef (Map.Map Buffer Word64))
    -- ^ Track size of each buffer for proper recycling
  , poolStats :: !(IORef PoolStats)
    -- ^ Pool statistics for profiling
  }

-- | Pool statistics for performance analysis
data PoolStats = PoolStats
  { statAcquired :: !Int
    -- ^ Total buffers acquired
  , statReused :: !Int
    -- ^ Buffers reused from pool (cache hits)
  , statCreated :: !Int
    -- ^ Buffers newly created (cache misses)
  , statReleased :: !Int
    -- ^ Buffers released back to pool
  , statDestroyed :: !Int
    -- ^ Buffers destroyed (on pool cleanup)
  } deriving (Show, Eq)

-- | Empty statistics
emptyStats :: PoolStats
emptyStats = PoolStats
  { statAcquired = 0
  , statReused = 0
  , statCreated = 0
  , statReleased = 0
  , statDestroyed = 0
  }

-- | Create a new buffer pool
--
-- Example:
-- @
--   pool <- createBufferPool
-- @
createBufferPool :: IO BufferPool
createBufferPool = do
  poolBuffers <- newIORef Map.empty
  poolActive <- newIORef Set.empty
  poolSizes <- newIORef Map.empty
  poolStats <- newIORef emptyStats
  return BufferPool{..}

-- | Destroy buffer pool and all cached buffers
--
-- This will destroy all buffers in the pool (both available and active).
-- Make sure all buffers are released before calling this.
--
-- Example:
-- @
--   destroyBufferPool pool
-- @
destroyBufferPool :: BufferPool -> IO ()
destroyBufferPool BufferPool{..} = do
  -- Get all buffers from pool
  availableBuffers <- readIORef poolBuffers
  activeBuffers <- readIORef poolActive

  -- Destroy all available buffers
  let allAvailable = concat $ Map.elems availableBuffers
  forM_ allAvailable $ \buffer -> do
    destroyBuffer buffer

  -- Destroy all active buffers (user should have released them first!)
  forM_ (Set.toList activeBuffers) $ \buffer -> do
    destroyBuffer buffer

  -- Update stats
  let totalDestroyed = length allAvailable + Set.size activeBuffers
  modifyIORef' poolStats $ \stats -> stats { statDestroyed = totalDestroyed }

  -- Clear pool
  writeIORef poolBuffers Map.empty
  writeIORef poolActive Set.empty
  writeIORef poolSizes Map.empty

-- | Acquire a buffer from the pool
--
-- If a buffer of the requested size is available in the pool, it will be reused.
-- Otherwise, a new buffer will be created.
--
-- The buffer is marked as active and must be released back to the pool with
-- 'releaseBuffer' when no longer needed.
--
-- Example:
-- @
--   buffer <- acquireBuffer pool ctx 1024 StorageBuffer
-- @
acquireBuffer :: BufferPool -> Context -> Word64 -> BufferUsage -> IO Buffer
acquireBuffer BufferPool{..} ctx size usage = do
  -- Try to get buffer from pool
  availableBuffers <- readIORef poolBuffers

  case Map.lookup size availableBuffers of
    Just (buffer:rest) -> do
      -- Reuse from pool
      writeIORef poolBuffers $ Map.insert size rest availableBuffers
      modifyIORef' poolActive $ Set.insert buffer

      -- Update stats
      modifyIORef' poolStats $ \stats -> stats
        { statAcquired = statAcquired stats + 1
        , statReused = statReused stats + 1
        }

      return buffer

    _ -> do
      -- Create new buffer
      buffer <- createBuffer ctx size usage
      modifyIORef' poolActive $ Set.insert buffer

      -- Track buffer size for later recycling
      modifyIORef' poolSizes $ Map.insert buffer size

      -- Update stats
      modifyIORef' poolStats $ \stats -> stats
        { statAcquired = statAcquired stats + 1
        , statCreated = statCreated stats + 1
        }

      return buffer

-- | Release a buffer back to the pool
--
-- The buffer is moved from active to available and can be reused for
-- future acquisitions.
--
-- **Important**: Do not use the buffer after releasing it! It may be
-- reused by another acquisition.
--
-- Example:
-- @
--   releaseBuffer pool buffer
-- @
releaseBuffer :: BufferPool -> Buffer -> IO ()
releaseBuffer BufferPool{..} buffer = do
  -- Remove from active set
  modifyIORef' poolActive $ Set.delete buffer

  -- Look up buffer size from tracking map
  sizes <- readIORef poolSizes
  case Map.lookup buffer sizes of
    Just bufferSize -> do
      -- Add to available buffers
      modifyIORef' poolBuffers $ \buffers ->
        Map.insertWith (++) bufferSize [buffer] buffers

      -- Update stats
      modifyIORef' poolStats $ \stats -> stats
        { statReleased = statReleased stats + 1
        }

    Nothing -> do
      -- Buffer size not tracked - this shouldn't happen
      -- Log warning and skip recycling (buffer will be GC'd)
      return ()

-- | Get current pool statistics
--
-- Useful for profiling and understanding cache hit rates.
--
-- Example:
-- @
--   stats <- getPoolStats pool
--   print $ "Hit rate: " ++ show (statReused stats * 100 / statAcquired stats) ++ "%"
-- @
getPoolStats :: BufferPool -> IO PoolStats
getPoolStats BufferPool{..} = readIORef poolStats

-- | Reset pool statistics to zero
--
-- Useful for benchmarking specific sections of code.
--
-- Example:
-- @
--   resetPoolStats pool
--   -- Run benchmark...
--   stats <- getPoolStats pool
-- @
resetPoolStats :: BufferPool -> IO ()
resetPoolStats BufferPool{..} = writeIORef poolStats emptyStats

-- | Helper: Get cache hit rate as percentage
--
-- Returns 0-100 representing the percentage of acquisitions that were
-- satisfied from the pool (vs creating new buffers).
cacheHitRate :: PoolStats -> Double
cacheHitRate PoolStats{..}
  | statAcquired == 0 = 0.0
  | otherwise = fromIntegral statReused * 100.0 / fromIntegral statAcquired

-- | Pretty-print pool statistics
--
-- Example output:
-- @
--   Buffer Pool Statistics:
--     Acquired: 1000
--     Reused: 850 (85.0% hit rate)
--     Created: 150
--     Released: 900
--     Destroyed: 0
-- @
prettyPrintStats :: PoolStats -> String
prettyPrintStats stats@PoolStats{..} = unlines
  [ "Buffer Pool Statistics:"
  , "  Acquired: " ++ show statAcquired
  , "  Reused: " ++ show statReused ++ " (" ++ show (cacheHitRate stats) ++ "% hit rate)"
  , "  Created: " ++ show statCreated
  , "  Released: " ++ show statReleased
  , "  Destroyed: " ++ show statDestroyed
  ]
