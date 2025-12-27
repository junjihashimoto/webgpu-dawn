{-|
Module      : WGSL.Analyze
Description : Static analysis for GPU shader performance bounds
Copyright   : (c) 2025
License     : BSD3

Roofline model analysis for WebGPU kernels.
Estimates compute intensity and performance bounds.

= Roofline Model

The Roofline model helps determine if a kernel is:
- **Compute Bound**: Limited by GPU's FLOPs capability
- **Memory Bound**: Limited by memory bandwidth

= Metrics

- **FLOPs**: Floating-point operations
- **Bytes**: Memory transfers (loads + stores)
- **Arithmetic Intensity**: FLOPs / Bytes (operations per byte)

= Performance Bounds

- Peak Compute: ~10-20 TFLOPS (typical GPU)
- Peak Bandwidth: ~200-400 GB/s (typical GPU)
- Roofline Threshold: ~25-100 FLOPs/Byte

If intensity < threshold: Memory bound
If intensity > threshold: Compute bound

= Example

@
  let analysis = analyzeMatMul 4096 F32
  print $ rooflineReport analysis
  -- Output:
  -- Compute: 137.4 GFLOPs
  -- Memory: 201.3 MB
  -- Arithmetic Intensity: 682.7 FLOPs/Byte
  -- Status: COMPUTE BOUND
@
-}

module WGSL.Analyze
  ( -- * Analysis Types
    KernelAnalysis(..)
  , PerformanceBound(..)
    -- * Analysis Functions
  , analyzeMatMul
  , analyzeLinear
  , analyzeAttention
    -- * Roofline Model
  , rooflineReport
  , theoreticalLimit
    -- * Hardware Models
  , GPUModel(..)
  , defaultGPU
  , appleM1GPU
  , appleM3GPU
  ) where

import Graphics.WebGPU.Dawn.Types (NumType(..))
import Text.Printf (printf)

-- | Performance bound classification
data PerformanceBound
  = ComputeBound     -- ^ Limited by compute throughput
  | MemoryBound      -- ^ Limited by memory bandwidth
  | Balanced         -- ^ Near the roofline threshold
  deriving (Show, Eq)

-- | Kernel analysis results
data KernelAnalysis = KernelAnalysis
  { kernelName :: !String
  , totalFLOPs :: !Double        -- ^ Total floating-point operations
  , totalBytes :: !Double        -- ^ Total memory transfers (bytes)
  , arithmeticIntensity :: !Double  -- ^ FLOPs / Byte
  , performanceBound :: !PerformanceBound
  , theoreticalTimeMs :: !Double  -- ^ Theoretical minimum time (ms)
  } deriving (Show, Eq)

-- | GPU hardware model
data GPUModel = GPUModel
  { gpuName :: !String
  , peakFLOPs :: !Double      -- ^ TFLOPS (10^12 FLOPs/s)
  , peakBandwidth :: !Double  -- ^ GB/s (10^9 Bytes/s)
  , rooflineThreshold :: !Double  -- ^ FLOPs/Byte threshold
  } deriving (Show, Eq)

-- | Generic GPU model (conservative estimates)
defaultGPU :: GPUModel
defaultGPU = GPUModel
  { gpuName = "Generic GPU"
  , peakFLOPs = 10.0  -- 10 TFLOPS
  , peakBandwidth = 200.0  -- 200 GB/s
  , rooflineThreshold = 50.0  -- FLOPs/Byte
  }

-- | Apple M1 GPU (8-core)
appleM1GPU :: GPUModel
appleM1GPU = GPUModel
  { gpuName = "Apple M1 (8-core GPU)"
  , peakFLOPs = 2.6  -- ~2.6 TFLOPS FP32
  , peakBandwidth = 68.25  -- ~68 GB/s
  , rooflineThreshold = 38.0  -- FLOPs/Byte
  }

-- | Apple M3 GPU (10-core)
appleM3GPU :: GPUModel
appleM3GPU = GPUModel
  { gpuName = "Apple M3 (10-core GPU)"
  , peakFLOPs = 3.5  -- ~3.5 TFLOPS FP32
  , peakBandwidth = 100.0  -- ~100 GB/s
  , rooflineThreshold = 35.0  -- FLOPs/Byte
  }

-- | Get size of numeric type in bytes
sizeOfNumType :: NumType -> Int
sizeOfNumType F16 = 2
sizeOfNumType F32 = 4
sizeOfNumType F64 = 8
sizeOfNumType I32 = 4
sizeOfNumType U32 = 4
sizeOfNumType _ = 4  -- Default to 4 bytes

-- | Analyze matrix multiplication kernel
-- MatMul [M, K] x [K, N] -> [M, N]
analyzeMatMul :: Int -> NumType -> KernelAnalysis
analyzeMatMul n dtype =
  let m = n
      k = n
      -- Compute: 2 * M * N * K FLOPs (multiply-add for each output element)
      flops = 2.0 * fromIntegral (m * n * k)

      -- Memory: Load A (M*K), Load B (K*N), Store C (M*N)
      elemSize = fromIntegral $ sizeOfNumType dtype
      loadA = fromIntegral (m * k) * elemSize
      loadB = fromIntegral (k * n) * elemSize
      storeC = fromIntegral (m * n) * elemSize
      bytes = loadA + loadB + storeC

      intensity = flops / bytes
      gpu = defaultGPU
      bound = classifyBound intensity (rooflineThreshold gpu)
      theoreticalMs = theoreticalLimit flops bytes gpu

  in KernelAnalysis
       { kernelName = "MatMul " ++ show n ++ "x" ++ show n
       , totalFLOPs = flops
       , totalBytes = bytes
       , arithmeticIntensity = intensity
       , performanceBound = bound
       , theoreticalTimeMs = theoreticalMs
       }

-- | Analyze linear layer (matrix-vector multiplication)
-- Linear: [N, D] x [D] -> [N]
analyzeLinear :: Int -> Int -> NumType -> KernelAnalysis
analyzeLinear n d dtype =
  let -- Compute: 2 * N * D FLOPs
      flops = 2.0 * fromIntegral (n * d)

      -- Memory: Load matrix (N*D), Load vector (D), Store output (N)
      elemSize = fromIntegral $ sizeOfNumType dtype
      loadMatrix = fromIntegral (n * d) * elemSize
      loadVector = fromIntegral d * elemSize
      storeOutput = fromIntegral n * elemSize
      bytes = loadMatrix + loadVector + storeOutput

      intensity = flops / bytes
      gpu = defaultGPU
      bound = classifyBound intensity (rooflineThreshold gpu)
      theoreticalMs = theoreticalLimit flops bytes gpu

  in KernelAnalysis
       { kernelName = "Linear " ++ show n ++ "x" ++ show d
       , totalFLOPs = flops
       , totalBytes = bytes
       , arithmeticIntensity = intensity
       , performanceBound = bound
       , theoreticalTimeMs = theoreticalMs
       }

-- | Analyze attention kernel (simplified)
-- Attention: [B, N, D] -> [B, N, D]
analyzeAttention :: Int -> Int -> Int -> NumType -> KernelAnalysis
analyzeAttention b n d dtype =
  let -- Simplified: Q@K^T (B*N*N*D), Softmax (B*N*N), Score@V (B*N*N*D)
      flops = 2.0 * fromIntegral (b * n * n * d + b * n * n + b * n * n * d)

      -- Memory: Load Q,K,V, Store output
      elemSize = fromIntegral $ sizeOfNumType dtype
      loadQKV = 3.0 * fromIntegral (b * n * d) * elemSize
      storeOutput = fromIntegral (b * n * d) * elemSize
      bytes = loadQKV + storeOutput

      intensity = flops / bytes
      gpu = defaultGPU
      bound = classifyBound intensity (rooflineThreshold gpu)
      theoreticalMs = theoreticalLimit flops bytes gpu

  in KernelAnalysis
       { kernelName = "Attention " ++ show b ++ "x" ++ show n ++ "x" ++ show d
       , totalFLOPs = flops
       , totalBytes = bytes
       , arithmeticIntensity = intensity
       , performanceBound = bound
       , theoreticalTimeMs = theoreticalMs
       }

-- | Classify performance bound based on arithmetic intensity
classifyBound :: Double -> Double -> PerformanceBound
classifyBound intensity threshold
  | intensity < threshold * 0.8 = MemoryBound
  | intensity > threshold * 1.2 = ComputeBound
  | otherwise = Balanced

-- | Calculate theoretical performance limit
theoreticalLimit :: Double -> Double -> GPUModel -> Double
theoreticalLimit flops bytes gpu =
  let -- Time limited by compute (ms)
      computeTimeMs = (flops / (peakFLOPs gpu * 1e12)) * 1000.0

      -- Time limited by memory (ms)
      memoryTimeMs = (bytes / (peakBandwidth gpu * 1e9)) * 1000.0

      -- Actual limit is the maximum (bottleneck)
  in max computeTimeMs memoryTimeMs

-- | Generate Roofline analysis report
rooflineReport :: KernelAnalysis -> String
rooflineReport analysis = unlines
  [ "=== Roofline Analysis: " ++ kernelName analysis ++ " ==="
  , ""
  , printf "[Compute]  %.2f GFLOPs" (totalFLOPs analysis / 1e9)
  , printf "[Memory]   %.2f MB" (totalBytes analysis / 1e6)
  , printf "[Intensity] %.2f FLOPs/Byte" (arithmeticIntensity analysis)
  , ""
  , "[Status]    " ++ boundDescription (performanceBound analysis)
  , printf "[Theoretical Limit] %.2f ms" (theoreticalTimeMs analysis)
  , ""
  , optimizationHint (performanceBound analysis)
  ]

-- | Description of performance bound
boundDescription :: PerformanceBound -> String
boundDescription ComputeBound = "COMPUTE BOUND (limited by GPU FLOPs)"
boundDescription MemoryBound = "MEMORY BOUND (limited by bandwidth)"
boundDescription Balanced = "BALANCED (near roofline threshold)"

-- | Optimization hint based on bound
optimizationHint :: PerformanceBound -> String
optimizationHint ComputeBound = unlines
  [ "[Optimization Hints]"
  , "  • Use lower precision (FP16 instead of FP32)"
  , "  • Leverage GPU tensor cores if available"
  , "  • Optimize compute kernels (loop unrolling, etc.)"
  ]
optimizationHint MemoryBound = unlines
  [ "[Optimization Hints]"
  , "  • Reduce memory transfers (kernel fusion)"
  , "  • Use shared memory / workgroup local memory"
  , "  • Increase data reuse (tiling, blocking)"
  , "  • Optimize memory access patterns (coalescing)"
  ]
optimizationHint Balanced = unlines
  [ "[Optimization Hints]"
  , "  • Balance compute and memory optimizations"
  , "  • Profile to identify actual bottleneck"
  , "  • Consider algorithm-level improvements"
  ]
