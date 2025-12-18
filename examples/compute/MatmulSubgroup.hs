{-|
Example: Matrix Multiplication using Subgroups with F16/F32 (Case 12 from run.cpp)

This implements the optimized subgroup matrix multiply with loop unrolling,
matching run.cpp case 12 exactly. Supports verification mode with small f32 matrices.
-}

module Main where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V
import System.Clock (Clock(..), getTime, diffTimeSpec, toNanoSecs)
import System.Environment (lookupEnv)
import Text.Printf (printf)

-- CPU reference implementation for matrix multiplication
-- matA is (m, k), matB is (n, k) transposed (column-major)
cpuMatmul :: V.Vector Float -> V.Vector Float -> Int -> Int -> Int -> V.Vector Float
cpuMatmul matA matB m k n = V.generate (m * n) $ \idx ->
  let row = idx `div` n
      col = idx `mod` n
      -- Dot product: A[row,:] * B[col,:]
      dotProd = sum [ (matA V.! (row * k + i)) * (matB V.! (col * k + i)) | i <- [0..k-1] ]
  in dotProd

-- Check if two vectors are close (for floating point comparison)
isClose :: V.Vector Float -> V.Vector Float -> Float -> (Bool, Int, Float, Float)
isClose v1 v2 tolerance =
  let diffs = V.zipWith (\a b -> abs (a - b)) v1 v2
      maxDiff = V.maximum diffs
      maxIdx = V.maxIndex diffs
      allClose = V.all (<= tolerance) diffs
  in (allClose, maxIdx, maxDiff, v1 V.! maxIdx)

-- Generate loop-unrolled shader code matching run.cpp's loopUnrolling()
-- precision can be "f16" or "f32"
subgroupMatmulShader :: String -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> String
subgroupMatmulShader precision _m k n tm tn lid0 lid1 = unlines $
  (if precision == "f16" then [ "enable f16;", "" ] else []) ++
  [ "enable chromium_experimental_subgroup_matrix;"
  , "diagnostic (off, chromium.subgroup_matrix_uniformity);"
  , ""
  , "@group(0) @binding(0) var<storage, read_write>  A: array<" ++ precision ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write>  B: array<" ++ precision ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write>  C: array<" ++ precision ++ ">;"
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
  , "  var Ax: array<subgroup_matrix_left<" ++ precision ++ ", 8, 8>, " ++ show tm ++ ">;"
  , "  var Bx: array<subgroup_matrix_right<" ++ precision ++ ", 8, 8>, " ++ show tn ++ ">;"
  , ""
  , "  // " ++ show tm ++ "x" ++ show tn ++ " accumulators (8x8 each)"
  , "  var accxx: array<subgroup_matrix_result<" ++ precision ++ ", 8, 8>, " ++ show tm ++ " * " ++ show tn ++ ">;"
  , ""
  ] ++
  -- Initialize Ax (unrolled)
  [ "  Ax[" ++ show i ++ "] = subgroup_matrix_left<" ++ precision ++ ", 8, 8>(0);" | i <- [0..tm-1] ] ++
  [ "  " ] ++
  -- Initialize Bx (unrolled)
  [ "  Bx[" ++ show i ++ "] = subgroup_matrix_right<" ++ precision ++ ", 8, 8>(0);" | i <- [0..tn-1] ] ++
  [ "  " ] ++
  -- Initialize accxx (unrolled)
  concat [ [ "  accxx[" ++ show i ++ "+" ++ show j ++ "*" ++ show tm ++ "] = subgroup_matrix_result<" ++ precision ++ ", 8, 8>(0);"
           | j <- [0..tn-1] ] ++ ["  "] | i <- [0..tm-1] ] ++
  [ "  "
  , ""
  , "  for (var k: u32 = 0u; k < " ++ show k ++ "; k = k + 8u) {"
  , "    workgroupBarrier();"
  ] ++
  -- Load Ax (unrolled)
  [ "    Ax[" ++ show i ++ "] = subgroupMatrixLoad<subgroup_matrix_left<" ++ precision ++ ",8,8>>(&A, baseA + k + 8u * " ++ show k ++ " * " ++ show i ++ ", false, " ++ show k ++ ");"
  | i <- [0..tm-1] ] ++
  [ "    " ] ++
  [ "" ] ++
  -- Load Bx (unrolled)
  [ "    Bx[" ++ show i ++ "] = subgroupMatrixLoad<subgroup_matrix_right<" ++ precision ++ ",8,8>>(&B, baseB + k * " ++ show n ++ " + 8u * " ++ show i ++ ", false, " ++ show n ++ ");"
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
  -- Check for verification mode
  verifyMode <- lookupEnv "VERIFY"
  let isVerify = case verifyMode of
        Just "1" -> True
        Just "true" -> True
        _ -> False

  if isVerify
    then runVerificationMode
    else runBenchmarkMode

runVerificationMode :: IO ()
runVerificationMode = evalContT $ do
  liftIO $ putStrLn "=== Subgroup Matrix Multiplication (F32 Verification Mode) ==="
  liftIO $ putStrLn "Testing with 128x128 matrices for CPU/GPU comparison"
  liftIO $ putStrLn ""

  -- Small matrices for verification
  let m = 128
      k = 128
      n = 128
      tm = 4
      tn = 2  -- Reduced to fit in 128x128
      lid0 = 32
      lid1 = 2

  liftIO $ putStrLn $ "Matrix dimensions: " ++ show m ++ "x" ++ show k ++ " * " ++ show n ++ "x" ++ show k
  liftIO $ putStrLn $ "Tile configuration: TM=" ++ show tm ++ ", TN=" ++ show tn ++ ", LID0=" ++ show lid0 ++ ", LID1=" ++ show lid1
  liftIO $ putStrLn ""

  -- Initialize test data with simple values for easy verification
  let matA = V.generate (m * k) (\i -> fromIntegral (i `mod` 10) / 10.0) :: V.Vector Float
      -- matBOrig is (N, K) - each row is a weight vector
      matBOrig = V.generate (n * k) (\i -> fromIntegral ((i + 5) `mod` 10) / 10.0) :: V.Vector Float
      -- Transpose B from (N, K) row-major to (N, K) column-major (which is same as (K, N) row-major)
      -- This matches run.cpp's transpose operation
      matB = V.generate (n * k) (\i ->
        let row = i `div` n  -- row in transposed (K, N) layout
            col = i `mod` n  -- col in transposed (K, N) layout
        in matBOrig V.! (col * k + row)) :: V.Vector Float

  liftIO $ putStrLn $ "Matrix A (M=" ++ show m ++ ", K=" ++ show k ++ ") row-major:"
  liftIO $ putStrLn $ "  First 5: " ++ show (V.take 5 matA)
  liftIO $ putStrLn $ "  A[0,0:5] = " ++ show (V.slice 0 5 matA)
  liftIO $ putStrLn $ "Matrix B (N=" ++ show n ++ ", K=" ++ show k ++ ") transposed (column-major):"
  liftIO $ putStrLn $ "  First 5: " ++ show (V.take 5 matB)
  liftIO $ putStrLn $ "  B[0,0:5] = " ++ show (V.slice 0 5 matB)
  liftIO $ putStrLn $ "  B is stored as (N,K) but represents transposed data"
  liftIO $ putStrLn ""

  -- Compute CPU reference using original (non-transposed) B
  liftIO $ putStrLn "Computing CPU reference..."
  let cpuResult = cpuMatmul matA matBOrig m k n
  liftIO $ putStrLn $ "CPU result (M=" ++ show m ++ ", N=" ++ show n ++ "):"
  liftIO $ putStrLn $ "  First 5: " ++ show (V.take 5 cpuResult)
  liftIO $ putStrLn $ "  C[0,0:5] = " ++ show (V.slice 0 5 cpuResult)
  liftIO $ putStrLn ""

  liftIO $ putStrLn "Creating context with subgroup features..."
  ctx <- createContextWithFeatures
    ["allow_unsafe_apis"]
    [FeatureShaderF16, FeatureSubgroups, FeatureChromiumExperimentalSubgroupMatrix]
  liftIO $ putStrLn "✓ Context created with subgroup features"

  liftIO $ putStrLn "Creating GPU tensors..."
  tensorA <- createTensorWithData ctx (Shape [m * k]) matA
  tensorB <- createTensorWithData ctx (Shape [n * k]) matB
  tensorC <- createTensor ctx (Shape [m * n]) F32
  liftIO $ putStrLn "✓ Tensors created"

  liftIO $ putStrLn "Compiling subgroup shader (F32)..."
  let shaderCode = subgroupMatmulShader "f32" m k n tm tn lid0 lid1
  code <- createKernelCode shaderCode
  liftIO $ putStrLn "✓ Shader compiled"

  let numWorkgroupsX = (m + 8 * tm - 1) `div` (8 * tm)
      numWorkgroupsY = (n + 8 * tn * lid1 - 1) `div` (8 * tn * lid1)

  liftIO $ putStrLn $ "Number of workgroups: (" ++ show numWorkgroupsX ++ ", " ++ show numWorkgroupsY ++ ", 1)"
  kernel <- createKernel ctx code [tensorA, tensorB, tensorC]
            (WorkgroupSize numWorkgroupsX numWorkgroupsY 1)
  liftIO $ putStrLn "✓ Kernel compiled"

  liftIO $ putStrLn "Dispatching kernel..."
  liftIO $ dispatchKernel ctx kernel
  liftIO $ putStrLn "✓ Kernel completed"

  liftIO $ putStrLn "Reading GPU results..."
  gpuResult <- liftIO $ fromGPU ctx tensorC (m * n) :: ContT r IO (V.Vector Float)
  liftIO $ putStrLn $ "GPU result (M=" ++ show m ++ ", N=" ++ show n ++ "):"
  liftIO $ putStrLn $ "  First 5: " ++ show (V.take 5 gpuResult)
  liftIO $ putStrLn $ "  C[0,0:5] = " ++ show (V.slice 0 5 gpuResult)
  liftIO $ putStrLn ""

  -- Manual check of first element
  let a00 = V.slice 0 k matA  -- First row of A
      b00 = V.slice 0 k matBOrig  -- First row of original B
      dotProduct = V.sum $ V.zipWith (*) a00 b00
  liftIO $ putStrLn $ "Manual verification of C[0,0]:"
  liftIO $ putStrLn $ "  A[0,:] dot BOrig[0,:] = " ++ show dotProduct
  liftIO $ putStrLn $ "  CPU result C[0,0] = " ++ show (cpuResult V.! 0)
  liftIO $ putStrLn $ "  GPU result C[0,0] = " ++ show (gpuResult V.! 0)
  liftIO $ putStrLn ""

  -- Compare results
  let tolerance = 1e-3  -- Allow small floating point differences
      (match, maxIdx, maxDiff, gpuVal) = isClose cpuResult gpuResult tolerance

  liftIO $ if match
    then putStrLn "✓ VERIFICATION PASSED: CPU and GPU results match!"
    else do
      putStrLn "✗ VERIFICATION FAILED: CPU and GPU results differ!"
      printf "  Maximum difference: %.6f at index %d\n" maxDiff maxIdx
      printf "  CPU value: %.6f, GPU value: %.6f\n" (cpuResult V.! maxIdx) gpuVal

  -- Resources automatically cleaned up by ContT!

runBenchmarkMode :: IO ()
runBenchmarkMode = evalContT $ do
  liftIO $ putStrLn "=== Subgroup Matrix Multiplication (F16 Benchmark Mode) ==="
  liftIO $ putStrLn "Testing chromium_experimental_subgroup_matrix - Case 12 from run.cpp"
  liftIO $ putStrLn ""

  -- Matching run.cpp case 12 exactly
  let m = 4096
      k = 4096
      n = 8192
      tm = 4
      tn = 8
      lid0 = 32
      lid1 = 2
      nIter = 30

  liftIO $ putStrLn $ "Matrix dimensions: " ++ show m ++ "x" ++ show k ++ " * " ++ show n ++ "x" ++ show k
  liftIO $ putStrLn $ "Tile configuration: TM=" ++ show tm ++ ", TN=" ++ show tn ++ ", LID0=" ++ show lid0 ++ ", LID1=" ++ show lid1
  liftIO $ putStrLn ""

  -- Initialize test data
  let matA = V.generate (m * k) (\i -> fromIntegral (i `mod` 10)) :: V.Vector Float
      matB = V.generate (n * k) (\i -> fromIntegral (i `mod` 10)) :: V.Vector Float

  liftIO $ putStrLn $ "Matrix A: first 5: " ++ show (V.take 5 matA)
  liftIO $ putStrLn $ "Matrix B: first 5: " ++ show (V.take 5 matB)
  liftIO $ putStrLn ""

  liftIO $ putStrLn "Creating context with subgroup features..."
  ctx <- createContextWithFeatures
    ["allow_unsafe_apis"]
    [FeatureShaderF16, FeatureSubgroups, FeatureChromiumExperimentalSubgroupMatrix]
  liftIO $ putStrLn "✓ Context created with subgroup features"

  liftIO $ putStrLn "Creating GPU tensors..."
  tensorA <- createTensorWithData ctx (Shape [m * k]) matA
  tensorB <- createTensorWithData ctx (Shape [n * k]) matB
  tensorC <- createTensor ctx (Shape [m * n]) F32
  liftIO $ putStrLn "✓ Tensors created"

  liftIO $ putStrLn "Compiling subgroup shader with loop unrolling..."
  let shaderCode = subgroupMatmulShader "f16" m k n tm tn lid0 lid1
  code <- createKernelCode shaderCode
  liftIO $ putStrLn "✓ Shader compiled"

  let numWorkgroupsX = (m + 8 * tm - 1) `div` (8 * tm)
      numWorkgroupsY = (n + 8 * tn * lid1 - 1) `div` (8 * tn * lid1)

  liftIO $ putStrLn $ "Workgroup size: (" ++ show lid0 ++ ", " ++ show lid1 ++ ", 1)"
  liftIO $ putStrLn $ "Number of workgroups: (" ++ show numWorkgroupsX ++ ", " ++ show numWorkgroupsY ++ ", 1)"
  kernel <- createKernel ctx code [tensorA, tensorB, tensorC]
            (WorkgroupSize numWorkgroupsX numWorkgroupsY 1)
  liftIO $ putStrLn "✓ Kernel compiled"

  liftIO $ putStrLn $ "Dispatching kernel " ++ show nIter ++ " times..."

  -- Run nIter times and collect timings
  times <- liftIO $ sequence $ replicate nIter $ do
    startTime <- getTime Monotonic
    dispatchKernel ctx kernel
    endTime <- getTime Monotonic
    let elapsed = diffTimeSpec startTime endTime
        elapsedNs = toNanoSecs elapsed
    return $ fromIntegral elapsedNs / 1.0e9 :: IO Double

  liftIO $ putStrLn $ "✓ All " ++ show nIter ++ " runs completed"

  -- Calculate statistics
  let totalOps = 2 * fromIntegral (m * n * k) :: Double
      avgTime = sum times / fromIntegral nIter
      minTime = minimum times
      maxTime = maximum times
      avgGflops = (totalOps / avgTime) / 1.0e9
      minGflops = (totalOps / maxTime) / 1.0e9
      maxGflops = (totalOps / minTime) / 1.0e9

  liftIO $ putStrLn ""
  liftIO $ putStrLn "===================================================================="
  liftIO $ printf "Execution Time: (M = %d, K = %d, N = %d) x %d iterations\n" m k n nIter
  liftIO $ printf "%.1f milliseconds / dispatch ~ %.2f GFLOPS\n" (avgTime * 1000) avgGflops
  liftIO $ putStrLn "===================================================================="
  liftIO $ putStrLn ""
  liftIO $ printf "Average execution time: %.4f ms\n" (avgTime * 1000)
  liftIO $ printf "Min execution time: %.4f ms (%.2f GFLOPS)\n" (minTime * 1000) maxGflops
  liftIO $ printf "Max execution time: %.4f ms (%.2f GFLOPS)\n" (maxTime * 1000) minGflops
  liftIO $ putStrLn ""

  liftIO $ putStrLn "Reading results..."
  result <- liftIO $ fromGPU ctx tensorC (m * n) :: ContT r IO (V.Vector Float)
  liftIO $ putStrLn $ "Result: first 5: " ++ show (V.take 5 result)
  liftIO $ putStrLn $ "Result: last 5: " ++ show (V.drop (m * n - 5) result)

  -- Resources automatically cleaned up by ContT!
  liftIO $ putStrLn ""
  liftIO $ putStrLn "✓ Subgroup matmul completed successfully on GPU!"
