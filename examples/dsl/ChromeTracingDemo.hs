{-# LANGUAGE DataKinds #-}

{-|
Chrome Tracing Profiling Demo

This example demonstrates:
1. Creating mock ProfileEvent data
2. Converting to Chrome Tracing format
3. Writing JSON trace files for visualization

Usage:
  cabal run chrome-tracing-demo
  # Then open chrome://tracing and load gpu_profile.json
-}

module Main where

import WGSL.Profile (ProfileEvent(..))
import WGSL.Profile.Trace

main :: IO ()
main = do
  putStrLn "=== Chrome Tracing Profiling Demo ==="
  putStrLn ""

  -- Create mock profiling events to demonstrate trace export
  -- In a real application, these would come from actual GPU profiling
  let events =
        [ ProfileEvent
            { eventName_ = "kernel_matmul"
            , startTime = 1000000  -- 1ms in nanoseconds
            , endTime = 3500000    -- 3.5ms
            , eventDurationMs = 2.5
            }
        , ProfileEvent
            { eventName_ = "kernel_reduce"
            , startTime = 4000000  -- 4ms
            , endTime = 5200000    -- 5.2ms
            , eventDurationMs = 1.2
            }
        , ProfileEvent
            { eventName_ = "kernel_copy"
            , startTime = 5500000  -- 5.5ms
            , endTime = 6000000    -- 6ms
            , eventDurationMs = 0.5
            }
        , ProfileEvent
            { eventName_ = "kernel_matmul"
            , startTime = 7000000  -- 7ms
            , endTime = 9500000    -- 9.5ms
            , eventDurationMs = 2.5
            }
        , ProfileEvent
            { eventName_ = "kernel_reduce"
            , startTime = 10000000  -- 10ms
            , endTime = 11200000    -- 11.2ms
            , eventDurationMs = 1.2
            }
        ]

  putStrLn "Created mock profiling events:"
  mapM_ print events
  putStrLn ""

  -- Convert to Chrome Tracing format
  putStrLn "Converting to Chrome Tracing format..."
  let trace = profileEventsToTrace events

  putStrLn $ "  Total events: " ++ show (length $ traceEvents trace)
  putStrLn $ "  Display unit: " ++ displayTimeUnit trace
  putStrLn ""

  -- Export to JSON file
  putStrLn "Writing to gpu_profile.json..."
  writeChromeTrace "gpu_profile.json" events

  putStrLn ""
  putStrLn "✅ Done!"
  putStrLn ""
  putStrLn "Next steps:"
  putStrLn "1. Open Chrome browser"
  putStrLn "2. Navigate to chrome://tracing"
  putStrLn "3. Click 'Load' button"
  putStrLn "4. Select gpu_profile.json"
  putStrLn "5. You'll see a timeline visualization of GPU kernels"
  putStrLn ""
  putStrLn "Features to explore in chrome://tracing:"
  putStrLn "  • WASD keys to navigate the timeline"
  putStrLn "  • Click events to see details (duration, name)"
  putStrLn "  • Identify overlapping operations and bottlenecks"
  putStrLn "  • Measure time between events"
  putStrLn ""
