{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE NumericUnderscores #-}

{-|
Module      : WGSL.Profile
Description : GPU profiling infrastructure using timestamp queries
Copyright   : (c) 2025
License     : BSD3

GPU timing support for WebGPU compute kernels using timestamp queries.
Provides high-precision measurements for identifying performance bottlenecks.
-}

module WGSL.Profile
  ( Profiler(..)  -- Export with all fields for low-level access
  , ProfileEvent(..)
  , createProfiler
  , destroyProfiler
  , resolveProfiler
  ) where

import Foreign
import Foreign.C.Types
import Foreign.C.String (peekCString)
import Data.Word (Word32, Word64)
import Graphics.WebGPU.Dawn.Internal
import Control.Exception (bracket)

-- | A profiler holds GPU timestamp query resources
data Profiler = Profiler
  { profContext :: !Context
  , profQuerySet :: !QuerySet
  , profResolveBuffer :: !Buffer
  , profMaxEvents :: !Int
  , profEventNames :: ![String]
  } deriving (Eq)

-- | A single profiling event with name, timestamps, and duration
data ProfileEvent = ProfileEvent
  { eventName_ :: !String      -- ^ Event name
  , startTime :: !Word64       -- ^ Start timestamp in nanoseconds
  , endTime :: !Word64         -- ^ End timestamp in nanoseconds
  , eventDurationMs :: !Double -- ^ Duration in milliseconds
  } deriving (Show, Eq, Ord)

-- | Create a profiler capable of recording up to maxEvents timestamp pairs
-- Each event needs 2 timestamps (start, end), so we allocate 2 * maxEvents queries
createProfiler :: Context -> Int -> IO Profiler
createProfiler ctx maxEvents = do
  -- Need 2 timestamps per event (start + end)
  let queryCount = fromIntegral (maxEvents * 2)

  -- Create query set for timestamp queries (type = 2 = GPU_QUERY_TYPE_TIMESTAMP)
  alloca $ \errPtr -> do
    poke errPtr (GPUError 0 nullPtr)

    querySet <- c_createQuerySet ctx 2 queryCount errPtr

    hasErr <- c_hasError errPtr
    if hasErr /= 0
      then do
        errMsg <- c_getLastErrorMessage errPtr >>= peekCString
        error $ "Failed to create query set: " ++ errMsg
      else do
        -- Create buffer to hold query results (8 bytes per timestamp)
        let bufferSize = fromIntegral (queryCount * 8)
        resolveBuffer <- c_createQueryBuffer ctx bufferSize errPtr

        hasErr2 <- c_hasError errPtr
        if hasErr2 /= 0
          then do
            c_destroyQuerySet querySet
            errMsg <- c_getLastErrorMessage errPtr >>= peekCString
            error $ "Failed to create query buffer: " ++ errMsg
          else
            return $ Profiler
              { profContext = ctx
              , profQuerySet = querySet
              , profResolveBuffer = resolveBuffer
              , profMaxEvents = maxEvents
              , profEventNames = []
              }

-- | Destroy profiler and free GPU resources
destroyProfiler :: Profiler -> IO ()
destroyProfiler prof = do
  c_destroyQuerySet (profQuerySet prof)
  c_releaseBuffer (profResolveBuffer prof)

-- | Resolve timestamp queries and return events with durations in milliseconds
-- Assumes queries were written in pairs: (start0, end0, start1, end1, ...)
resolveProfiler :: Profiler -> IO [ProfileEvent]
resolveProfiler prof = do
  let ctx = profContext prof
      numQueries = profMaxEvents prof * 2

  -- Read back timestamp data (uint64 nanoseconds)
  allocaArray numQueries $ \timestampsPtr -> do
    alloca $ \errPtr -> do
      poke errPtr (GPUError 0 nullPtr)

      c_readQueryBuffer ctx (profResolveBuffer prof) timestampsPtr
                       (fromIntegral numQueries) errPtr

      hasErr <- c_hasError errPtr
      if hasErr /= 0
        then do
          errMsg <- c_getLastErrorMessage errPtr >>= peekCString
          error $ "Failed to read query buffer: " ++ errMsg
        else do
          timestamps <- peekArray numQueries timestampsPtr

          -- Convert pairs of timestamps to events
          let events = convertTimestamps (profEventNames prof) timestamps
          return events

-- | Convert array of timestamps into profile events
-- timestamps = [start0, end0, start1, end1, ...]
convertTimestamps :: [String] -> [Word64] -> [ProfileEvent]
convertTimestamps names timestamps = go names 0 timestamps
  where
    go :: [String] -> Int -> [Word64] -> [ProfileEvent]
    go [] _ _ = []
    go _ _ [] = []
    go _ _ [_] = []  -- Need pairs
    go (name:ns) idx (t1:t2:ts) =
      let durationNs = fromIntegral (t2 - t1) :: Double
          durationMs = durationNs / 1_000_000.0  -- Convert nanoseconds to milliseconds
          event = ProfileEvent
            { eventName_ = name
            , startTime = t1
            , endTime = t2
            , eventDurationMs = durationMs
            }
      in event : go ns (idx + 1) ts
    go _ _ _ = []

-- Additional helper: Set event names after creation
setEventNames :: Profiler -> [String] -> Profiler
setEventNames prof names = prof { profEventNames = names }
