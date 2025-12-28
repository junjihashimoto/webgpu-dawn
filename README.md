# webgpu-dawn

High-level, type-safe Haskell bindings to Google's [Dawn WebGPU](https://dawn.googlesource.com/dawn) implementation.

This library enables **portable GPU computing** with a **Production-Ready DSL** designed for high-throughput inference (e.g., LLMs), targeting **300 TPS (Tokens Per Second)** performance.

---

## ⚡ Core Design Principles

To achieve high performance and type safety, this library adheres to the following strict patterns:

1. **Type-Safe Monadic DSL:** No raw strings. We use `ShaderM` for composability and type safety.
2. **Natural Math & HOAS:** Standard operators (`+`, `*`) and Higher-Order Abstract Syntax (HOAS) for loops (`loop ... $ \i -> ...`).
3. **Profile-Driven:** Performance tuning is based on Roofline Analysis.
4. **Async Execution:** Prefer `AsyncPipeline` to hide CPU latency and maximize GPU occupancy.
5. **Hardware Acceleration:** Mandatory use of **Subgroup Operations** and **F16** precision for heavy compute (MatMul/Reduction).

---

## 🏎️ Performance & Profiling

We utilize a **Profile-Driven Development (PDD)** workflow to maximize throughput.

### 1. Standard Benchmarks & Roofline Analysis

Run the optimized benchmark to determine TFLOPS and check the Roofline classification (Compute vs Memory Bound).

```bash
# Run 2D Block-Tiling MatMul Benchmark (FP32)
cabal run bench-optimized-matmul -- --size 4096 --iters 50

```

**Output Example:**

```text
[Compute]  137.4 GFLOPs
[Memory]   201.3 MB
[Status]   COMPUTE BOUND (limited by GPU FLOPs)
[Hint]     Use F16 and Subgroup Operations to break the roofline.

```

### 2. Visual Profiling (Chrome Tracing)

Generate a trace file to visualize CPU/GPU overlap and kernel duration.

```bash
cabal run bench-optimized-matmul -- --size 4096 --trace

```

* **Load:** Open `chrome://tracing` or [ui.perfetto.dev](https://ui.perfetto.dev)
* **Analyze:** Import `trace.json` to identify gaps between kernel executions (CPU overhead).

### 3. Debugging

Use the GPU printf-style debug buffer to inspect values inside kernels.

```haskell
-- In DSL:
debugPrintF "intermediate_val" val

```

---

## 🚀 Quick Start

### 1. High-Level API (Data Parallelism)

Zero boilerplate. Ideal for simple `map`/`reduce` tasks.

```haskell
import WGSL.API
import qualified Data.Vector.Storable as V

main :: IO ()
main = withContext $ \ctx -> do
  input  <- toGPU ctx (V.fromList [1..100] :: V.Vector Float)
  result <- gpuMap (\x -> x * 2.0 + 1.0) input
  out    <- fromGPU' result
  print out

```

### 2. Core DSL (Explicit Control)

Required for tuning **Shared Memory**, **Subgroups**, and **F16**.

```haskell
import WGSL.DSL

shader :: ShaderM ()
shader = do
  input  <- declareInputBuffer "in" (TArray 1024 TF16)
  output <- declareOutputBuffer "out" (TArray 1024 TF16)
   
  -- HOAS Loop: Use lambda argument 'i', NOT string "i"
  loop 0 1024 1 $ \i -> do
    val <- readBuffer input i
    -- f16 literals for 2x throughput
    let res = val * litF16 2.0 + litF16 1.0
    writeBuffer output i res

```

---

## 📚 DSL Syntax Cheatsheet

### Types & Literals

| Haskell Type | WGSL Type | Literal Constructor | Note |
| --- | --- | --- | --- |
| `Exp F32` | `f32` | `litF32 1.0` or `1.0` | Standard float |
| `Exp F16` | `f16` | `litF16 1.0` | Half precision (Fast!) |
| `Exp I32` | `i32` | `litI32 1` or `1` | Signed int |
| `Exp U32` | `u32` | `litU32 1` | Unsigned int |
| `Exp Bool_` | `bool` | `litBool True` | Boolean |

**Casting Helpers:** `i32(e)`, `u32(e)`, `f32(e)`, `f16(e)`

### Control Flow (HOAS)

```haskell
-- For Loop
loop start end step $ \i -> do ...

-- If Statement
if_ (val > 10.0) 
    (do ... {- then block -} ...) 
    (do ... {- else block -} ...)

-- Barrier
barrier  -- workgroupBarrier()

```

---

## 🧩 Kernel Fusion

For maximum performance, fuse multiple operations (`Load` -> `Calc` -> `Store`) into a single kernel to reduce global memory traffic.

```haskell
import WGSL.Kernel

-- Fuse: Load -> Process -> Store
let pipeline = loadK inBuf >>> mapK (* 2.0) >>> mapK relu >>> storeK outBuf

-- Execute inside shader
unKernel pipeline i

```

---

## 📚 Architecture & Modules

### Execution Model (Latency Hiding)

To maximize GPU occupancy, encoding is separated from submission.

* **`WGSL.Async.Pipeline`**: Use for main loops. Allows CPU to encode Token `N+1` while GPU processes Token `N`.
* **`WGSL.Execute`**: Low-level synchronous execution (primarily for debugging).

### Module Guide

| Feature | Module | Description |
| --- | --- | --- |
| **Subgroup Ops** | `WGSL.DSL` | `subgroupMatrixLoad`, `mma`, `subgroupMatrixStore` |
| **F16 Math** | `WGSL.DSL` | `litF16`, `vec4<f16>` for 2x throughput |
| **Structs** | `WGSL.Struct` | `Generic` derivation for `std430` layout compliance |
| **Analysis** | `WGSL.Analyze` | Roofline analysis logic |

---

## 📦 Installation

Pre-built Dawn binaries are downloaded automatically during installation.

```bash
cabal install webgpu-dawn

```

---

## License

MIT License - see [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## Acknowledgments

* **Dawn (Google):** Core WebGPU runtime.
* **gpu.cpp (Answer.AI):** High-level C++ API wrapper inspiration.
* **GLFW:** Window management.

## Contact

Maintainer: Junji Hashimoto [junji.hashimoto@gmail.com](mailto:junji.hashimoto@gmail.com)
