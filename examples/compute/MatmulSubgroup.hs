{-|
Example: Matrix Multiplication using Subgroups with F16 (Case 12 from run.cpp)

This implements the optimized subgroup matrix multiply with loop unrolling,
matching run.cpp case 12 exactly.
-}

module Main where

import Graphics.WebGPU.Dawn
import qualified Data.Vector.Storable as V
import System.Clock (Clock(..), getTime, diffTimeSpec, toNanoSecs)
import Text.Printf (printf)

-- Generate loop-unrolled shader code matching run.cpp's loopUnrolling()
subgroupMatmulShaderF16 :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> String
subgroupMatmulShaderF16 _m k n tm tn lid0 lid1 = unlines $
  [ "enable f16;"
  , ""
  , "enable chromium_experimental_subgroup_matrix;"
  , "diagnostic (off, chromium.subgroup_matrix_uniformity);"
  , ""
  , "@group(0) @binding(0) var<storage, read_write>  A: array<f16>;"
  , "@group(0) @binding(1) var<storage, read_write>  B: array<f16>;"
  , "@group(0) @binding(2) var<storage, read_write>  C: array<f16>;"
  , ""
  , "@compute @workgroup_size(" ++ show lid0 ++ ", " ++ show lid1 ++ ", 1)"
  , "fn main(@builtin(workgroup_id) wg: vec3<u32>,"
  , "        @builtin(local_invocation_id) localID : vec3<u32>) {"
  , ""
  , "  let rowStart: u32 = wg.x * 8u * " ++ show tm ++ ";"
  , "  let colStart: u32 = (wg.y * " ++ show lid1 ++ " + localID.y)  * 8u * " ++ show tn ++ ";"
  , ""
  , "  let baseA: u32 = rowStart * " ++ show k ++ ";"
  , "  let baseB: u32 = colStart;"
  , "  let cBase: u32 = rowStart * " ++ show n ++ " + colStart;"
  , ""
  , "  var Ax: array<subgroup_matrix_left<f16, 8, 8>, " ++ show tm ++ ">;"
  , "  var Bx: array<subgroup_matrix_right<f16, 8, 8>, " ++ show tn ++ ">;"
  , ""
  , "  // " ++ show tm ++ "x" ++ show tn ++ " accumulators (8x8 each)"
  , "  var accxx: array<subgroup_matrix_result<f16, 8, 8>, " ++ show tm ++ " * " ++ show tn ++ ">;"
  , ""
  ] ++
  -- Initialize Ax (unrolled)
  [ "  Ax[" ++ show i ++ "] = subgroup_matrix_left<f16, 8, 8>(0);" | i <- [0..tm-1] ] ++
  [ "  " ] ++
  -- Initialize Bx (unrolled)
  [ "  Bx[" ++ show i ++ "] = subgroup_matrix_right<f16, 8, 8>(0);" | i <- [0..tn-1] ] ++
  [ "  " ] ++
  -- Initialize accxx (unrolled)
  concat [ [ "  accxx[" ++ show i ++ "+" ++ show j ++ "*" ++ show tm ++ "] = subgroup_matrix_result<f16, 8, 8>(0);"
           | j <- [0..tn-1] ] ++ ["  "] | i <- [0..tm-1] ] ++
  [ "  "
  , ""
  , "  for (var k: u32 = 0u; k < " ++ show k ++ "; k = k + 8u) {"
  , "    workgroupBarrier();"
  ] ++
  -- Load Ax (unrolled)
  [ "    Ax[" ++ show i ++ "] = subgroupMatrixLoad<subgroup_matrix_left<f16,8,8>>(&A, baseA + k + 8u * " ++ show k ++ " * " ++ show i ++ ", false, " ++ show k ++ ");"
  | i <- [0..tm-1] ] ++
  [ "    " ] ++
  [ "" ] ++
  -- Load Bx (unrolled)
  [ "    Bx[" ++ show i ++ "] = subgroupMatrixLoad<subgroup_matrix_right<f16,8,8>>(&B, baseB + k * " ++ show n ++ " + 8u * " ++ show i ++ ", false, " ++ show n ++ ");"
  | i <- [0..tn-1] ] ++
  [ "    " ] ++
  [ "" ] ++
  -- Multiply-accumulate (unrolled)
  concat [ [ "    accxx[" ++ show j ++ "*" ++ show tm ++ " + " ++ show i ++ "] = subgroupMatrixMultiplyAccumulate(Ax[" ++ show i ++ "], Bx[" ++ show j ++ "], accxx[" ++ show j ++ "*" ++ show tm ++ " + " ++ show i ++ "]);"
           | i <- [0..tm-1] ] ++ ["      "] | j <- [0..tn-1] ] ++
  [ "    "
  , "  }"
  , ""
  , "  workgroupBarrier();"
  ] ++
  -- Store results (unrolled)
  concat [ [ "  subgroupMatrixStore(&C, cBase + " ++ show i ++ " * 8u * " ++ show n ++ " + 8u * " ++ show j ++ ", accxx[" ++ show j ++ "*" ++ show tm ++ " + " ++ show i ++ "], false, " ++ show n ++ ");"
           | j <- [0..tn-1] ] ++ ["    "] | i <- [0..tm-1] ] ++
  [ "}"
  ]

main :: IO ()
main = do
  putStrLn "=== Subgroup Matrix Multiplication (F16 with Loop Unrolling) ==="
  putStrLn "Testing chromium_experimental_subgroup_matrix - Case 12 from run.cpp"
  putStrLn ""

  -- Matching run.cpp case 12 exactly
  let m = 4096
      k = 4096
      n = 8192
      tm = 4
      tn = 8
      lid0 = 32
      lid1 = 2
      nIter = 30

  putStrLn $ "Matrix dimensions: " ++ show m ++ "x" ++ show k ++ " * " ++ show n ++ "x" ++ show k
  putStrLn $ "Tile configuration: TM=" ++ show tm ++ ", TN=" ++ show tn ++ ", LID0=" ++ show lid0 ++ ", LID1=" ++ show lid1
  putStrLn ""

  -- Initialize test data - using simple pattern for now
  let matA = V.generate (m * k) (\i -> fromIntegral (i `mod` 10)) :: V.Vector Float
      matB = V.generate (n * k) (\i -> fromIntegral (i `mod` 10)) :: V.Vector Float

  putStrLn $ "Matrix A: first 5: " ++ show (V.take 5 matA)
  putStrLn $ "Matrix B: first 5: " ++ show (V.take 5 matB)
  putStrLn ""

  putStrLn "Creating context with subgroup features..."
  withContextFeatures
    ["allow_unsafe_apis"]  -- Enable experimental features
    [FeatureShaderF16, FeatureSubgroups, FeatureChromiumExperimentalSubgroupMatrix]
    $ \ctx -> do
      putStrLn "✓ Context created with subgroup features"

      putStrLn "Creating GPU tensors..."
      tensorA <- createTensorWithData ctx (Shape [m * k]) matA
      tensorB <- createTensorWithData ctx (Shape [n * k]) matB
      tensorC <- createTensor ctx (Shape [m * n]) F32
      putStrLn "✓ Tensors created"

      putStrLn "Compiling subgroup shader with loop unrolling..."
      let shaderCode = subgroupMatmulShaderF16 m k n tm tn lid0 lid1
      code <- createKernelCode shaderCode
      putStrLn "✓ Shader compiled"

      let numWorkgroupsX = (m + 8 * tm - 1) `div` (8 * tm)
          numWorkgroupsY = (n + 8 * tn * lid1 - 1) `div` (8 * tn * lid1)

      putStrLn $ "Workgroup size: (" ++ show lid0 ++ ", " ++ show lid1 ++ ", 1)"
      putStrLn $ "Number of workgroups: (" ++ show numWorkgroupsX ++ ", " ++ show numWorkgroupsY ++ ", 1)"
      kernel <- compileKernel ctx code [tensorA, tensorB, tensorC]
                (WorkgroupSize numWorkgroupsX numWorkgroupsY 1)
      putStrLn "✓ Kernel compiled"

      putStrLn $ "Dispatching kernel " ++ show nIter ++ " times..."

      -- Run nIter times and collect timings (matching run.cpp)
      times <- sequence $ replicate nIter $ do
        startTime <- getTime Monotonic
        dispatchKernel ctx kernel
        endTime <- getTime Monotonic
        let elapsed = diffTimeSpec startTime endTime
            elapsedNs = toNanoSecs elapsed
        return $ fromIntegral elapsedNs / 1.0e9 :: IO Double

      putStrLn $ "✓ All " ++ show nIter ++ " runs completed"

      -- Calculate statistics
      let totalOps = 2 * fromIntegral (m * n * k) :: Double
          avgTime = sum times / fromIntegral nIter
          minTime = minimum times
          maxTime = maximum times
          avgGflops = (totalOps / avgTime) / 1.0e9
          minGflops = (totalOps / maxTime) / 1.0e9
          maxGflops = (totalOps / minTime) / 1.0e9

      putStrLn ""
      putStrLn "===================================================================="
      printf "Execution Time: (M = %d, K = %d, N = %d) x %d iterations\n" m k n nIter
      printf "%.1f milliseconds / dispatch ~ %.2f GFLOPS\n" (avgTime * 1000) avgGflops
      putStrLn "===================================================================="
      putStrLn ""
      printf "Average execution time: %.4f ms\n" (avgTime * 1000)
      printf "Min execution time: %.4f ms (%.2f GFLOPS)\n" (minTime * 1000) maxGflops
      printf "Max execution time: %.4f ms (%.2f GFLOPS)\n" (maxTime * 1000) minGflops
      putStrLn ""

      putStrLn "Reading results..."
      result <- fromGPU ctx tensorC (m * n) :: IO (V.Vector Float)
      putStrLn $ "Result: first 5: " ++ show (V.take 5 result)
      putStrLn $ "Result: last 5: " ++ show (V.drop (m * n - 5) result)

      destroyTensor tensorA
      destroyTensor tensorB
      destroyTensor tensorC
      destroyKernel kernel
      destroyKernelCode code

      putStrLn ""
      putStrLn "✓ Subgroup matmul completed successfully on GPU!"
