{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : WGSL.Profile.Trace
Description : Chrome Tracing format export for GPU profiling
Copyright   : (c) 2025
License     : MIT

Export GPU profiling data in Chrome Tracing JSON format for visualization.

= Usage Pattern

1. Run profiled kernels and collect ProfileEvent list
2. Convert to Chrome Tracing JSON
3. Write to file
4. Open in chrome://tracing

= Example

@
  -- Run profiled operations
  result <- executeShaderWithProfiling ctx "matmul" shader tensors wgSize
  let events = profiledEvents result

  -- Export to Chrome Tracing format
  writeChromeTrace "profile.json" events

  -- Then open chrome://tracing and load profile.json
@

= Chrome Tracing Format

The format uses JSON with the following structure:

@
{
  "traceEvents": [
    {
      "name": "kernel_name",
      "cat": "gpu",
      "ph": "X",
      "ts": 1234567.890,
      "dur": 123.456,
      "pid": 1,
      "tid": 1
    }
  ],
  "displayTimeUnit": "ns"
}
@

Where:
- name: Event name
- cat: Category (always "gpu" for our use case)
- ph: Phase ("X" for complete events with duration)
- ts: Timestamp in microseconds (start time)
- dur: Duration in microseconds
- pid: Process ID (always 1)
- tid: Thread ID (always 1 for sequential GPU operations)
-}

module WGSL.Profile.Trace
  ( ChromeTraceEvent(..)
  , ChromeTrace(..)
  , profileEventToTraceEvent
  , profileEventsToTrace
  , writeChromeTrace
  , chromeTraceJSON
  ) where

import WGSL.Profile (ProfileEvent(..))
import Data.Aeson (ToJSON(..), object, (.=), encode)
import qualified Data.ByteString.Lazy as BL
import Data.List (sort)

-- | A single event in Chrome Tracing format
data ChromeTraceEvent = ChromeTraceEvent
  { eventName :: !String      -- ^ Event name
  , eventCategory :: !String  -- ^ Category (e.g., "gpu")
  , eventPhase :: !String     -- ^ Phase type ("X" for complete events)
  , eventTimestamp :: !Double -- ^ Start timestamp in microseconds
  , eventDuration :: !Double  -- ^ Duration in microseconds
  , eventPid :: !Int          -- ^ Process ID
  , eventTid :: !Int          -- ^ Thread ID
  } deriving (Show, Eq)

instance ToJSON ChromeTraceEvent where
  toJSON e = object
    [ "name" .= eventName e
    , "cat" .= eventCategory e
    , "ph" .= eventPhase e
    , "ts" .= eventTimestamp e
    , "dur" .= eventDuration e
    , "pid" .= eventPid e
    , "tid" .= eventTid e
    ]

-- | Chrome Tracing document containing multiple events
data ChromeTrace = ChromeTrace
  { traceEvents :: ![ChromeTraceEvent]
  , displayTimeUnit :: !String
  } deriving (Show, Eq)

instance ToJSON ChromeTrace where
  toJSON t = object
    [ "traceEvents" .= traceEvents t
    , "displayTimeUnit" .= displayTimeUnit t
    ]

-- | Convert a ProfileEvent to ChromeTraceEvent
--
-- GPU timestamps are in nanoseconds, Chrome Tracing expects microseconds
profileEventToTraceEvent :: ProfileEvent -> ChromeTraceEvent
profileEventToTraceEvent event =
  ChromeTraceEvent
    { eventName = eventName_ event
    , eventCategory = "gpu"
    , eventPhase = "X"  -- Complete event (has duration)
    , eventTimestamp = fromIntegral (startTime event) / 1000.0  -- ns -> μs
    , eventDuration = fromIntegral (endTime event - startTime event) / 1000.0  -- ns -> μs
    , eventPid = 1
    , eventTid = 1
    }

-- | Convert list of ProfileEvents to ChromeTrace
--
-- Events are sorted by start time for better visualization
profileEventsToTrace :: [ProfileEvent] -> ChromeTrace
profileEventsToTrace events =
  let sortedEvents = sort events  -- ProfileEvent should have Ord instance based on startTime
      traceEvents_ = map profileEventToTraceEvent sortedEvents
  in ChromeTrace
       { traceEvents = traceEvents_
       , displayTimeUnit = "ns"
       }

-- | Generate Chrome Tracing JSON from ProfileEvents
chromeTraceJSON :: [ProfileEvent] -> BL.ByteString
chromeTraceJSON = encode . profileEventsToTrace

-- | Write Chrome Tracing JSON to file
--
-- The resulting file can be loaded in chrome://tracing for visualization
--
-- Example:
-- @
--   events <- runProfiledKernels
--   writeChromeTrace "gpu_profile.json" events
--   -- Then open chrome://tracing and load gpu_profile.json
-- @
writeChromeTrace :: FilePath -> [ProfileEvent] -> IO ()
writeChromeTrace filepath events = do
  let json = chromeTraceJSON events
  BL.writeFile filepath json
  putStrLn $ "Chrome trace written to " ++ filepath
  putStrLn $ "Open chrome://tracing and load the file to visualize"
