# webgpu-dawn Examples

This directory contains examples demonstrating the webgpu-dawn Haskell bindings for GPU computing and graphics.

## Directory Structure

```
examples/
├── dsl/         - Examples using the WGSL DSL (recommended)
├── compute/     - Examples using raw WGSL strings
└── graphics/    - Graphics examples using GLFW (requires -f glfw)
```

## Quick Start

### Compute Examples (DSL)

The **recommended** way to write GPU compute shaders in Haskell:

```bash
# Vector operations using the DSL
cabal run vector-add-dsl
cabal run real-vector-double-dsl

# Matrix multiplication using the DSL
cabal run matmul-dsl
cabal run shared-memory-reduction

# Advanced: Subgroup matrix operations
cabal run matmul-subgroup-dsl
```

See `dsl/README.md` for details on the WGSL DSL.

### Compute Examples (Raw WGSL)

Traditional approach using WGSL strings directly:

```bash
cabal run wgsl-vector-addition
cabal run gpu-vector-double
cabal run simple-integration
```

See `compute/README.md` for details.

### Graphics Examples

Windowed graphics applications (requires GLFW):

```bash
# Build with glfw support
cabal build -f glfw

# Run examples
cabal run -f glfw triangle-gui
cabal run -f glfw tetris-gui
cabal run -f glfw gltf-viewer
```

See `graphics/README.md` for details.

## Choosing an Approach

### Use the DSL (`examples/dsl/`) if you want:
- ✅ Type-safe GPU shader construction
- ✅ Compile-time checking
- ✅ Operator overloading (`+`, `*`, etc.)
- ✅ Haskell-native shader programming
- ✅ Phantom types for memory safety

**Example:**
```haskell
{-# LANGUAGE RebindableSyntax #-}
import WGSL.DSL

myShader :: ShaderM ()
myShader = do
  gid <- globalId
  let idx = vecX gid
  let input = globalBuffer "input" :: Ptr Storage (Array 256 F32)
  val <- readBuffer input idx
  writeBuffer output idx (val * litF32 2.0)
```

### Use raw WGSL (`examples/compute/`) if you want:
- ✅ Direct control over WGSL output
- ✅ Bleeding-edge GPU features
- ✅ Simpler mental model (just strings)
- ✅ Easier porting from existing WGSL code

**Example:**
```haskell
myShader :: String
myShader = unlines
  [ "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "    output[gid.x] = input[gid.x] * 2.0;"
  , "}"
  ]
```

## Learning Path

1. **Start here**: `dsl/VectorAddDSL.hs` or `compute/VectorAddition.hs`
   - Simple vector addition
   - Basic workflow

2. **Next**: `dsl/RealVectorDoubleDSL.hs` or `compute/GPUVectorDouble.hs`
   - Reading from buffers
   - Writing results

3. **Intermediate**: `dsl/MatmulDSL.hs`
   - Nested loops
   - Matrix operations

4. **Advanced**: `dsl/SharedMemoryReduction.hs`
   - Shared memory
   - Barriers
   - Parallel reduction

5. **Expert**: `dsl/MatmulSubgroupDSL.hs` or `compute/MatmulSubgroup.hs`
   - Subgroup operations
   - High-performance computing
   - Advanced GPU features

## Documentation

- **DSL Reference**: See `dsl/README.md` for WGSL DSL documentation
- **Raw WGSL**: See `compute/README.md` for traditional approach
- **Graphics**: See `graphics/README.md` for GLFW examples
- **Main README**: See `../README.md` for package overview

## Features by Example

| Feature | DSL Example | Raw WGSL Example |
|---------|------------|------------------|
| Basic compute | VectorAddDSL.hs | VectorAddition.hs |
| Buffer I/O | RealVectorDoubleDSL.hs | GPUVectorDouble.hs |
| Loops | MatmulDSL.hs | - |
| Shared memory | SharedMemoryReduction.hs | - |
| Subgroups | MatmulSubgroupDSL.hs | MatmulSubgroup.hs |
| Graphics | - | triangle-gui, tetris-gui |

## Platform Support

All compute examples work on:
- ✅ macOS (Metal backend)
- ✅ Linux (Vulkan backend)
- ✅ Windows (DirectX 12 backend)

Graphics examples require additional dependencies (GLFW, windowing system).
