# webgpu-dawn

High-level, type-safe Haskell bindings to Google's [Dawn WebGPU](https://dawn.googlesource.com/dawn) implementation.
This library enables **portable GPU computing** on macOS (Metal), Windows (DirectX 12), and Linux (Vulkan) with a **Production-Ready DSL**.

---

## 🤖 Context for AI Coding Assistants

**System Prompt:**
If you are an AI assistant writing code using this library, strictly adhere to the **"Phase 6 DSL"** patterns described below.

1. **NO Raw Strings:** Never use string concatenation for shaders. Use the monadic DSL.
2. **NO String Variables:** Use HOAS loops (`loop ... $ \i -> ...`), never string literals (`for_ "i" ...`).
3. **Natural Math:** Use standard operators (`+`, `*`) and cast helpers (`i32`, `f32`), not `Add`/`Mul` constructors.
4. **Safe Execution:** ALWAYS use `executeShaderNamed` with `AnyTensor`.

---

## 🚀 Quick Start

### 1. High-Level API (Data Parallelism)

Zero boilerplate. Use this for simple `map`/`reduce` tasks.

```haskell
import WGSL.API
import qualified Data.Vector.Storable as V

main = withContext $ \ctx -> do
  -- Transfer data
  input <- toGPU ctx (V.fromList [1..100] :: V.Vector Float)
  
  -- Automatically generates and executes compute kernels
  result <- gpuMap (\x -> x * 2.0 + 1.0) input
  
  out <- fromGPU' result
  print out

```

### 2. Core DSL (Explicit Control)

For tuning **Shared Memory**, **Subgroups**, and **F16**.

```haskell
import WGSL.DSL

shader :: ShaderM ()
shader = do
  -- Auto-binding: No manual @binding(n) indices needed
  input  <- declareInputBuffer "in" (TArray 1024 TF16)
  output <- declareOutputBuffer "out" (TArray 1024 TF16)
  
  -- HOAS Loop: Use lambda argument 'i', NOT string "i"
  loop 0 1024 1 $ \i -> do
    val <- readBuffer input i
    -- Natural Math: Use standard operators (+, *, /)
    let res = val * litF16 2.0 + litF16 1.0
    writeBuffer output i res

```

---

## 📚 DSL Reference (Cheatsheet)

### 1. Types & Literals

| Haskell Type | WGSL Type | Literal Constructor | Note |
| --- | --- | --- | --- |
| `Exp F32` | `f32` | `litF32 1.0` or `1.0` | Standard float |
| `Exp F16` | `f16` | `litF16 1.0` | Half precision (fast!) |
| `Exp I32` | `i32` | `litI32 1` or `1` | Signed int |
| `Exp U32` | `u32` | `litU32 1` | Unsigned int |
| `Exp Bool_` | `bool` | `litBool True` | Boolean |

**Casting Helpers:**

* `i32(e)`, `u32(e)`, `f32(e)`, `f16(e)`

### 2. Memory & Buffers

Buffers declared with `declare*` are automatically assigned binding indices in order.

```haskell
-- Global Buffers
input  <- declareInputBuffer "name" (TArray size type)
output <- declareOutputBuffer "name" (TArray size type)

-- Shared Memory (Workgroup)
shared <- shared (TArray size type)

-- Private Variables (Registers)
reg <- var type initialValue

-- Access Methods
val <- readBuffer buffer index    -- Load from Global
writeBuffer buffer index val      -- Store to Global
val <- readPtr ptr                -- Read Register/Shared
assign ptr val                    -- Write Register/Shared

```

### 3. Control Flow (HOAS)

Do not use string literals for variable names.

```haskell
-- For Loop
loop start end step $ \i -> do
  ...

-- If Statement
if_ (val > 10.0)
  (do ... then block ...)
  (do ... else block ...)

-- Barrier
barrier  -- workgroupBarrier()

```

### 4. Structs & Memory Layout

Use `GHC.Generics` to auto-generate `std430` compliant structs (handles padding automatically).

```haskell
{-# LANGUAGE DeriveGeneric #-}

data Particle = Particle 
  { pos :: Vec3 F32   -- Automatically aligned to 16 bytes
  , vel :: Vec3 F32 
  , mass :: F32
  } deriving (Generic, WGSLStorable)

-- In DSL (Access fields):
-- let p = ... :: Exp (Struct Particle)
-- let p_pos = p ^. "pos"

```

---

## 🧩 Kernel Fusion (Advanced)

For maximum performance, you can fuse multiple operations (`Load` -> `Calc` -> `Store`) into a single kernel to reduce global memory traffic.
This separates the **Definition** of a kernel from its **Execution**.

### 1. `Kernel` vs `ShaderM`

* **`ShaderM a`**: Imperative code generation (Effect).
* **`Kernel wX wY wZ input output`**: A composable wrapper around `ShaderM` (Function).

### 2. Defining Kernels

Use `Kernel` constructor for effectful ops, or `mapK` for pure math.

```haskell
import WGSL.Kernel

-- Pure Math Kernel
process :: Kernel 256 1 1 (Exp F32) (Exp F32)
process = mapK (* 2.0) >>> mapK (+ 1.0) >>> mapK relu

-- Effectful Kernel (Load)
loadK :: Ptr Storage (Array n F32) -> Kernel 256 1 1 (Exp I32) (Exp F32)
loadK buf = Kernel $ \idx -> readBuffer buf idx

```

### 3. Executing Kernels (Fusion)

Compose them with `>>>` and run them inside `ShaderM` using `unKernel`.

```haskell
mainShader :: ShaderM ()
mainShader = do
  inBuf  <- declareInputBuffer "in" ...
  outBuf <- declareOutputBuffer "out" ...

  -- Fuse: Load -> Process -> Store
  let pipeline = loadK inBuf >>> process >>> storeK outBuf
  
  -- Execute the fused pipeline
  loop 0 1024 1 $ \i -> do
    unKernel pipeline i  -- <--- Generates the fused WGSL code here

```

---

## 🏎️ Execution (Host Side)

**Always use `executeShaderNamed**`. It matches Haskell Tensors to WGSL Buffers by **Name**, preventing order-mismatch bugs.

```haskell
import Graphics.WebGPU.Dawn
import WGSL.Execute

main = do
  -- ... setup context ...
  
  executeShaderNamed ctx shaderModule
    [ ("in",  AnyTensor inputTensor)  -- F32 Tensor
    , ("out", AnyTensor outputTensor) -- F32 Tensor
    , ("params", AnyTensor paramTensor) -- I32 Tensor (Heterogeneous types allowed!)
    ]
    (WorkgroupSize 256 1 1)

```

---

## 🛠 Project Structure

* **`src/WGSL/API.hs`**: High-level `map`/`reduce` wrappers.
* **`src/WGSL/DSL.hs`**: Core operators, literals, and builder functions.
* **`src/WGSL/Monad.hs`**: State monad for AST generation.
* **`src/WGSL/Kernel.hs`**: Composable kernel fusion mechanism.
* **`src/WGSL/Struct.hs`**: Generic struct derivation and memory layout logic.
* **`src/WGSL/Execute.hs`**: Runtime execution and binding logic.

## 📦 Installation

```bash
cabal install webgpu-dawn

```

*(Pre-built Dawn binaries are downloaded automatically during installation)*

---

## License

MIT License - see [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## Acknowledgments

This project builds upon and includes code from:

* **Dawn (Google):** Core WebGPU runtime.
* **gpu.cpp (Answer.AI):** High-level C++ API wrapper.
* **GLFW:** Window management.

## Contact

Maintainer: Junji Hashimoto [junji.hashimoto@gmail.com](mailto:junji.hashimoto@gmail.com)
