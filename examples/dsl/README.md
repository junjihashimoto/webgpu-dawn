# WGSL DSL Examples

This directory contains examples that use the **WGSL DSL** (Domain Specific Language) for building GPU compute shaders programmatically.

## What is the WGSL DSL?

The WGSL DSL is a Haskell EDSL (Embedded Domain Specific Language) that allows you to:
- Build GPU shaders using monadic Haskell code
- Get compile-time type safety for GPU operations
- Use operator overloading for natural mathematical syntax
- Generate WGSL code from a typed AST (not string concatenation)

## Key Features

✓ **Type-safe expressions**: `Exp F32`, `Exp I32`, `Exp (Vec3 U32)`
✓ **Memory space tracking**: `Ptr Storage`, `Ptr Workgroup`, `Ptr Private`
✓ **Monadic construction**: `ShaderM` monad for building shaders
✓ **Control flow**: `if_`, `for_`, `while_`, `barrier`
✓ **Operator overloading**: `+`, `*`, `==`, `<` for GPU expressions
✓ **Subgroup operations**: Full support for subgroup matrix multiplication

## Examples

### VectorAddDSL.hs
Simple vector addition showing basic DSL usage.

### RealVectorDoubleDSL.hs
Demonstrates reading from buffers, arithmetic operations, and writing results.

### MatmulDSL.hs
Matrix multiplication using DSL with nested loops and accumulation.

### SharedMemoryReduction.hs
Parallel reduction using shared memory (workgroup memory) and barriers.

### MatmulSubgroupDSL.hs
Advanced example using subgroup matrix operations for high-performance matrix multiplication.

### MatmulSubgroupDSL2.hs
**IMPROVED** version of the subgroup matrix multiplication example demonstrating dsl2.md improvements:
- Automatic variable naming (no string manipulation)
- `staticFor` for compile-time unrolling
- Smart constructors hiding internal types
- High-level operation wrappers (`loadMatrix`, `storeMatrix`, `mma`)

See `DSL_IMPROVEMENTS.md` for detailed comparison!

### MatmulDSL3.hs ⭐ NEW ⭐
**PRODUCTION-READY** example demonstrating ALL dsl3.md production features:
- Multi-dimensional views for safe array indexing
- Kernel configuration integration
- Debug print support
- Optimized code generation

See `DSL3_IMPROVEMENTS.md` for complete feature guide!

## How to Run

```bash
cabal run vector-add-dsl
cabal run real-vector-double-dsl
cabal run matmul-dsl
cabal run shared-memory-reduction
cabal run matmul-subgroup-dsl
cabal run matmul-subgroup-dsl2  # dsl2.md improvements
cabal run matmul-dsl3  # ⭐ dsl3.md production features!
```

## DSL Modules

The DSL is implemented across several modules:
- `WGSL.AST` - Core AST types with phantom type safety
- `WGSL.Monad` - ShaderM monad for imperative shader construction
- `WGSL.DSL` - Operator overloading and type classes
- `WGSL.CodeGen` - Pretty-printer to generate WGSL strings
- `WGSL.Builder` - Helper functions for building complete shaders
- `WGSL.Execute` - Integration with webgpu-dawn execution

## Example Code Structure

```haskell
{-# LANGUAGE RebindableSyntax #-}

import WGSL.DSL

myShader :: ShaderM ()
myShader = do
  -- Get thread ID
  gid <- globalId
  let idx = vecX gid

  -- Reference storage buffers
  let input = globalBuffer "input" :: Ptr Storage (Array 256 F32)
  let output = globalBuffer "output" :: Ptr Storage (Array 256 F32)

  -- Read, compute, write
  val <- readBuffer input idx
  let result = val * litF32 2.0
  writeBuffer output idx result
```

This is **real DSL** - not string manipulation!

## DSL Improvements

The DSL has undergone two major improvement cycles:

### Phase 1: Code Clarity (dsl2.md) ✅
1. **Automatic variable naming** - No more `"var_" ++ show i`
2. **staticFor** - Explicit compile-time unrolling vs runtime loops
3. **Smart constructors** - Hide internal AST types from user code
4. **High-level wrappers** - `loadMatrix`, `storeMatrix`, `mma` for cleaner code

**See details**: `DSL_IMPROVEMENTS.md`
**Try the example**: `cabal run matmul-subgroup-dsl2`

### Phase 2: Production-Ready (dsl3.md) ✅
1. **Multi-dimensional views** - Type-safe 2D/3D array indexing with automatic stride calculation
2. **Constant folding** - AST optimization for cleaner generated WGSL
3. **Pipeline integration** - 🔧 Pending: Include metadata in shader definitions
4. **Debug print support** - 🔧 Pending: Printf-style GPU debugging

**See details**: `DSL3_IMPROVEMENTS.md`

**Example with views:**
```haskell
let viewA = makeView2D matA m k  -- Logical 2D view
val <- readView2D viewA row col  -- Safe, automatic offset calculation
```

**Example with optimization:**
```haskell
let wgsl = generateWGSLOptimized shaderModule  -- Optimized code generation
```
