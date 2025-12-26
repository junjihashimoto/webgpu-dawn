# Compute Examples (Raw WGSL)

This directory contains examples that use **raw WGSL strings** for GPU compute shaders.

## What is WGSL?

WGSL (WebGPU Shading Language) is the shader language for WebGPU. These examples demonstrate writing WGSL code directly as strings and executing them via the webgpu-dawn bindings.

## Comparison with DSL Examples

| Raw WGSL (this directory) | WGSL DSL (`examples/dsl/`) |
|---------------------------|----------------------------|
| Write shaders as strings  | Build shaders with Haskell code |
| String concatenation      | Type-safe AST construction |
| No compile-time checking  | Phantom types for safety |
| Direct WGSL control       | Operator overloading |

## Examples

### VectorAddition.hs
Simple WGSL shader for vector addition. Shows the basic workflow:
1. Create context
2. Write WGSL string
3. Compile kernel
4. Execute on GPU

### GPUVectorDouble.hs
Demonstrates doubling each element of a vector using raw WGSL.

### SimpleIntegration.hs
Shows integration between WGSL strings and the webgpu-dawn execution pipeline.

### VectorAdd.hs
Basic vector addition kernel.

### MatmulSubgroup.hs
**Advanced**: Matrix multiplication using Chrome's experimental subgroup matrix operations.
Shows how to use cutting-edge GPU features with raw WGSL:
- `enable chromium_experimental_subgroup_matrix;`
- `subgroup_matrix_left<f16, 8, 8>`
- `subgroupMatrixLoad`
- `subgroupMatrixMultiplyAccumulate`
- `subgroupMatrixStore`

This is the reference implementation that the DSL version (`MatmulSubgroupDSL.hs`) aims to replicate.

## How to Run

```bash
cabal run wgsl-vector-addition
cabal run gpu-vector-double
cabal run simple-integration
```

## Typical Raw WGSL Workflow

```haskell
import Graphics.WebGPU.Dawn
import Graphics.WebGPU.Dawn.Tensor
import Graphics.WebGPU.Dawn.Kernel

-- 1. Write WGSL as a string
myShader :: String
myShader = unlines
  [ "@group(0) @binding(0) var<storage, read> input: array<f32>;"
  , "@group(0) @binding(1) var<storage, read_write> output: array<f32>;"
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {"
  , "    let idx = global_id.x;"
  , "    output[idx] = input[idx] * 2.0;"
  , "}"
  ]

-- 2. Compile and execute
main :: IO ()
main = bracket createContext destroyContext $ \ctx -> do
  code <- createKernelCode myShader
  -- ... create tensors, kernel, dispatch ...
```

## When to Use Raw WGSL vs DSL

**Use Raw WGSL when:**
- You need direct control over WGSL output
- You're prototyping or translating existing WGSL code
- You want to use bleeding-edge features not yet in the DSL
- You prefer working with the actual shader language

**Use DSL (`examples/dsl/`) when:**
- You want type safety and compile-time checking
- You need to generate shaders programmatically
- You prefer working in Haskell
- You want operator overloading and high-level abstractions
