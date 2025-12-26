# webgpu-dawn

High-level Haskell bindings to Google's [Dawn WebGPU](https://dawn.googlesource.com/dawn) implementation.
This library enables **portable GPU computing** on macOS (Metal), Windows (DirectX 12), and Linux (Vulkan) with a **Production-Ready DSL**.

## 🚀 Quick Start

This library features a **Two-Layer Architecture**:

### 1. High-Level API (Like Accelerate)

For zero-boilerplate data parallelism. Use this for simple tasks.

```haskell
import WGSL.API
import qualified Data.Vector.Storable as V

main = withContext $ \ctx -> do
  -- Transfer data
  input <- toGPU ctx (V.fromList [1..100] :: V.Vector Float)
  
  -- No shader code needed!
  -- Automatically generates and executes compute kernels
  result <- gpuMap (\x -> x * 2.0 + 1.0) input
  
  -- Get result
  out <- fromGPU' result
  print out

```

### 2. Core DSL (For Performance Tuning)

For explicit control over **Shared Memory**, **Subgroups**, and **F16**. Use this for high-performance algorithms (e.g., LLM inference, Physics).

```haskell
import WGSL.DSL

-- Define Shader Logic
shader :: ShaderM ()
shader = do
  -- Auto-binding: No manual @binding(n) indices needed!
  input  <- declareInputBuffer "in" (TArray 1024 TF16)
  output <- declareOutputBuffer "out" (TArray 1024 TF16)
  
  -- Natural Math Syntax & HOAS Loops
  loop 0 1024 1 $ \i -> do
    val <- readBuffer input i
    -- Use F16 for 2x performance
    let res = val * litF16 2.0 + litF16 1.0
    writeBuffer output i res

```

---

## 💎 Key Features

* **🛡️ Type Safety & Safety Mechanisms**
* **Auto-Binding:** `declareInputBuffer` automatically assigns `@binding(n)` indices.
* **Named Execution:** `executeShaderNamed` matches buffers by name, preventing runtime order mismatch.
* **HOAS Loops:** No string-based variables (`loop ... $ \i -> ...`), preventing scope errors.


* **⚡ Zero-Overhead Abstractions**
* **Struct Generation:** `GHC.Generics` derives WGSL structs with correct `std430` padding automatically.
* **Subgroup Operations:** First-class support for `subgroupMatrixMultiply` (MMA) for matrix multiplication.
* **Typed Views:** `MatrixView` prevents row/column index swapping bugs.


* **🔧 Debugging**
* **`debugPrint`:** Insert printf-style debugging comments in shaders.
* **Validation:** Runtime checks for missing or extra buffers.



---

## 📚 DSL Reference for LLMs/Developers

**Note to AI Coding Assistants:** Please follow the "DO" patterns below for best results.

### ❌ Do NOT do this (Old Style):

```haskell
-- BAD: String manipulation and unsafe list-based execution
let src = "var x = " ++ show i ++ ";"
executeShader ctx src [t1, t2] ... -- Order matters! Dangerous!

```

### ✅ DO this (New Style):

```haskell
-- GOOD: Monadic DSL, Named Execution, and Auto-Binding
x <- var TF32 (litF32 0.0)
loop 0 10 1 $ \i -> do ...

-- Safe Execution by Name
executeShaderNamed ctx shader 
  [ ("input", AnyTensor t1)
  , ("output", AnyTensor t2) 
  ] ...

```

### ✅ Struct Definition (Best Practice):

```haskell
{-# LANGUAGE DeriveGeneric #-}

data Particle = Particle 
  { pos :: Vec3 F32
  , vel :: Vec3 F32 
  , mass :: F32
  } deriving (Generic, WGSLStorable)

-- Automatically handles 16-byte alignment for vec3 in std430 layouts!

```

---

## 🛠 Project Structure

* **`src/WGSL/API.hs`**: High-level `map`/`reduce` wrappers.
* **`src/WGSL/DSL.hs`**: Core operators, literals, and builder functions.
* **`src/WGSL/Struct.hs`**: Generic struct derivation and memory layout logic.
* **`src/WGSL/Execute.hs`**: Runtime execution and binding logic.
* **`src/WGSL/Kernel.hs`**: Composable kernel fusion mechanism.

## 📦 Installation

```bash
cabal install webgpu-dawn

```

*(Pre-built Dawn binaries are downloaded automatically during installation)*

---

## License

MIT License - see [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## Acknowledgments

This project builds upon and includes code from several open-source projects:

* **Dawn (Google):** Core WebGPU runtime.
* **gpu.cpp (Answer.AI):** High-level C++ API wrapper.
* **GLFW:** Window management for graphics examples.
* **WebGPU Specification:** W3C Standard.

See `cbits/THIRD_PARTY_LICENSES.md` for complete license texts.

## Links

* [Documentation (Hackage)](https://hackage.haskell.org/package/webgpu-dawn)
* [WebGPU Shading Language Spec](https://www.w3.org/TR/WGSL/)

## Contact

Maintainer: Junji Hashimoto [junji.hashimoto@gmail.com](mailto:junji.hashimoto@gmail.com)
