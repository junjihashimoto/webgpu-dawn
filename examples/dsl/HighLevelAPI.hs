{-|
High-Level API Example - The "Accelerate" Layer

This demonstrates the new high-level API that provides zero-boilerplate
GPU computing, similar to Haskell Accelerate.

Compare this example to the low-level DSL examples - notice how much
simpler it is!
-}

module Main where

import Prelude
import qualified Prelude as P
import Control.Exception (bracket)
import qualified Data.Vector.Storable as V

import WGSL.API
-- No need to import WGSL.DSL - WGSL.API re-exports what we need
-- and the Num instances work automatically

main :: IO ()
main = do
  putStrLn "=== High-Level API Example ==="
  putStrLn ""
  putStrLn "This demonstrates the TWO-LAYER architecture:"
  putStrLn "  Layer 1 (HIGH): Easy-to-use operations (this example)"
  putStrLn "  Layer 2 (CORE): Explicit control for performance tuning"
  putStrLn ""

  bracket createContext destroyContext P.$ \ctx -> do
    -- Create input data
    let inputData = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 :: Float]

    putStrLn "Input vector:"
    print inputData
    putStrLn ""

    -- Transfer to GPU
    gpuInput <- toGPU ctx inputData

    -- Example 1: Simple map - ZERO BOILERPLATE!
    putStrLn "Example 1: Double every element"
    putStrLn "  Code: gpuMap (*2.0) gpuInput"
    gpuDoubled <- gpuMap (*2.0) gpuInput
    resultDoubled <- fromGPU' gpuDoubled
    putStrLn "  Result:"
    print resultDoubled
    putStrLn ""

    -- Example 2: Chain operations
    putStrLn "Example 2: Multiply by 2, then add 1"
    putStrLn "  Code: gpuMap (+1.0) =<< gpuMap (*2.0) gpuInput"
    gpuChained <- gpuMap (+1.0) =<< gpuMap (*2.0) gpuInput
    resultChained <- fromGPU' gpuChained
    putStrLn "  Result:"
    print resultChained
    putStrLn ""

    -- Example 3: ZipWith (element-wise binary operation)
    putStrLn "Example 3: Element-wise addition of two vectors"
    putStrLn "  Code: gpuZipWith (+) gpuA gpuB"
    let inputB = V.fromList [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0 :: Float]
    gpuInputB <- toGPU ctx inputB
    gpuSum <- gpuZipWith (+) gpuInput gpuInputB
    resultSum <- fromGPU' gpuSum
    putStrLn "  Result:"
    print resultSum
    putStrLn ""

    -- Example 4: Parallel fold (sum)
    putStrLn "Example 4: Parallel sum (fold)"
    putStrLn "  Code: gpuFold (+) 0 gpuInput"
    sumResult <- gpuFold (\a b -> a + b) 0.0 gpuInput
    putStrLn P.$ "  Result: " P.++ show sumResult
    putStrLn ""

    putStrLn "BENEFITS OF HIGH-LEVEL API:"
    putStrLn "============================"
    putStrLn "• Zero boilerplate - no shader code needed"
    putStrLn "• Automatic buffer management"
    putStrLn "• Type-safe transformations"
    putStrLn "• Compiler-generated optimized shaders"
    putStrLn ""
    putStrLn "WHEN TO USE LOW-LEVEL DSL:"
    putStrLn "=========================="
    putStrLn "• Shared memory optimizations"
    putStrLn "• Subgroup operations (matrix multiply)"
    putStrLn "• Complex control flow"
    putStrLn "• Custom memory layouts"
    putStrLn ""
    putStrLn "Phase 7 (Task 3): High-Level API ✓ Complete!"
