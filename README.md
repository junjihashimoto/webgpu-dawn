# webgpu-dawn

High-level Haskell bindings to Google's [Dawn WebGPU](https://dawn.googlesource.com/dawn) implementation for GPU computing and graphics programming.

## Features

- **Type-safe GPU Computing**: Compile and run WGSL compute shaders with automatic resource management
- **Graphics Rendering**: Support for vertex/fragment shaders and render pipelines
- **Automatic Setup**: Dawn is downloaded and built automatically during package installation
- **Cross-platform**: Supports macOS (Metal), Linux (Vulkan), and Windows (D3D12)
- **Zero-copy Data Transfer**: Efficient CPU↔GPU data transfer using `vector` and `Storable`
- **Safe Resource Management**: Automatic cleanup with Haskell's resource management patterns

## Quick Start

### Installation

```bash
cabal update
cabal install webgpu-dawn
```

The first build will download and compile Dawn (~10-15 minutes). Subsequent builds are fast.

### Simple Example

```haskell
import Graphics.WebGPU.Dawn
import qualified Data.Vector.Storable as V

main :: IO ()
main = withContext $ \ctx -> do
  -- Create GPU tensors
  let a = V.fromList [1, 2, 3, 4] :: V.Vector Float
      b = V.fromList [5, 6, 7, 8] :: V.Vector Float
      shape = Shape [4]

  tensorA <- createTensorWithData ctx shape a
  tensorB <- createTensorWithData ctx shape b
  tensorC <- createTensor ctx shape F32

  -- Compile shader
  let shader = unlines
        [ "@group(0) @binding(0) var<storage, read> a: array<f32>;"
        , "@group(0) @binding(1) var<storage, read> b: array<f32>;"
        , "@group(0) @binding(2) var<storage, read_write> c: array<f32>;"
        , ""
        , "@compute @workgroup_size(256)"
        , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
        , "  c[gid.x] = a[gid.x] + b[gid.x];"
        , "}"
        ]

  code <- createKernelCode shader
  kernel <- compileKernel ctx code [tensorA, tensorB, tensorC]
                         (WorkgroupSize 1 1 1)

  -- Execute on GPU
  dispatchKernel ctx kernel

  -- Read results
  result <- fromGPU ctx tensorC 4
  print result  -- [6.0, 8.0, 10.0, 12.0]
```

## Examples

The package includes several examples demonstrating different use cases:

### GPU Computing Examples

- **VectorAdd.hs**: Basic element-wise vector addition
- **MatrixMultiply.hs**: Optimized matrix multiplication with performance comparison
- **Convolution.hs**: 2D image convolution with various filters (blur, edge detection, sharpen)

### Graphics Examples

- **Triangle.hs**: Basic triangle rendering with vertex colors
- **TexturedQuad.hs**: Texture mapping with transformation matrices
- **Cube3D.hs**: 3D rendering with Phong lighting model

### Hybrid Examples

- **ParticleSimulation.hs**: Physics simulation using compute shaders with graphical rendering

Run examples with:

```bash
cd examples/compute
cabal run VectorAdd
```

## API Overview

### Context Management

```haskell
withContext :: (Context -> IO a) -> IO a
createContext :: IO Context
destroyContext :: Context -> IO ()
```

### Tensor Operations

```haskell
-- Create tensors
createTensor :: Context -> Shape -> NumType -> IO Tensor
createTensorWithData :: TensorData a => Context -> Shape -> Vector a -> IO Tensor

-- Data transfer
toGPU :: TensorData a => Context -> Tensor -> Vector a -> IO ()
fromGPU :: TensorData a => Context -> Tensor -> Int -> IO (Vector a)

-- Supported types
instance TensorData Float
instance TensorData Double
instance TensorData Int32
instance TensorData Word32
```

### Kernel Compilation

```haskell
-- Create and configure shader code
createKernelCode :: String -> IO KernelCode
setWorkgroupSize :: KernelCode -> WorkgroupSize -> IO ()
setEntryPoint :: KernelCode -> String -> IO ()

-- Compile and execute
compileKernel :: Context -> KernelCode -> [Tensor] -> WorkgroupSize -> IO Kernel
dispatchKernel :: Context -> Kernel -> IO ()
```

### Types

```haskell
data Shape = Shape [Int]
data NumType = F16 | F32 | F64 | I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64
data WorkgroupSize = WorkgroupSize { workgroupX, workgroupY, workgroupZ :: Int }
```

## Configuration

### Environment Variables

- `DAWN_HOME`: Custom installation directory (default: `~/.cache/dawn`)
- `DAWN_VERSION`: Specific Dawn commit to use (default: tested commit)
- `DAWN_SKIP_BUILD`: Skip building Dawn (assumes system installation)

### Platform Support

| Platform | Backend | Status |
|----------|---------|--------|
| macOS (Apple Silicon) | Metal | ✅ Supported |
| macOS (Intel) | Metal | ✅ Supported |
| Linux (x86_64) | Vulkan | ✅ Supported |
| Windows | D3D12 | 🚧 Experimental |

## Architecture

```
webgpu-dawn
├── Setup.hs                  # Custom Cabal setup (downloads/builds Dawn)
├── cbits/
│   ├── gpu_wrapper.h         # C API header
│   ├── gpu_wrapper.c         # C helper functions
│   └── gpu_cpp_bridge.cpp    # C++ wrapper around gpu.cpp
├── src/Graphics/WebGPU/Dawn/
│   ├── Internal.hs           # Low-level FFI bindings
│   ├── Types.hs              # Type definitions
│   ├── Context.hs            # Context management
│   ├── Tensor.hs             # Tensor operations
│   └── Kernel.hs             # Kernel compilation/execution
└── examples/                 # Example programs
```

## Dependencies

### Build Dependencies

- CMake 3.14+
- Git
- C++17 compiler (clang++/g++/MSVC)
- Platform-specific:
  - macOS: Xcode Command Line Tools
  - Linux: Vulkan drivers (`libvulkan1`, `mesa-vulkan-drivers`)
  - Windows: Visual Studio 2019+

### Runtime Dependencies

None! The Dawn shared library is bundled with the package.

## Troubleshooting

### Dawn Build Fails

```bash
# Clean and rebuild
rm -rf ~/.cache/dawn
cabal clean
cabal configure
cabal build
```

### Linker Errors on macOS

The package automatically adds `-ld_classic` for macOS. If you still see errors:

```bash
export DAWN_SKIP_BUILD=1
# Install Dawn manually and ensure it's in your library path
```

### GPU Not Found

Ensure your system has compatible GPU drivers:

```bash
# Linux
sudo apt install vulkan-tools
vulkaninfo

# macOS
system_profiler SPDisplaysDataType
```

## Performance Tips

1. **Batch Operations**: Compile kernels once, dispatch multiple times
2. **Minimize Transfers**: Keep data on GPU when possible
3. **Workgroup Size**: Tune for your GPU (typically multiples of 32/64)
4. **Shared Memory**: Use workgroup-local storage for tiled algorithms

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Graphics pipeline API (render passes, vertex buffers)
- [ ] Texture and sampler support
- [ ] Async compute operations
- [ ] Window integration (GLFW/SDL bindings)
- [ ] More numeric types (F16 support)
- [ ] Profiling and debugging tools

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This project builds upon and includes code from several open-source projects:

### Dawn (Google)
- **Project**: [Dawn - Chrome's WebGPU implementation](https://dawn.googlesource.com/dawn)
- **License**: BSD 3-Clause License
- **Usage**: Core WebGPU runtime, automatically downloaded and built during installation
- **Copyright**: Copyright 2017-2024 The Dawn & Tint Authors

### gpu.cpp (Answer.AI)
- **Project**: [gpu.cpp - Minimal GPU compute library](https://github.com/AnswerDotAI/gpu.cpp)
- **License**: Apache License 2.0
- **Usage**: High-level C++ API wrapper included in `cbits/gpu.hpp`
- **Copyright**: Copyright (c) 2024 Answer.AI
- **Note**: This project uses gpu.hpp to provide a simplified interface to Dawn's native APIs

### GLFW (Optional)
- **Project**: [GLFW - Multi-platform library for OpenGL/Vulkan](https://github.com/glfw/glfw)
- **License**: zlib/libpng License
- **Usage**: Window management for graphics examples (when built with `-fglfw` flag)
- **Copyright**: Copyright (c) 2002-2006 Marcus Geelnard, Copyright (c) 2006-2019 Camilla Löwy

### WebGPU Specification
- **Project**: [WebGPU W3C Specification](https://www.w3.org/TR/webgpu/)
- **License**: W3C Software and Document License
- **Usage**: API design follows the WebGPU standard

See `cbits/THIRD_PARTY_LICENSES.md` for complete license texts.

### Special Thanks
- The Dawn team at Google for creating an excellent WebGPU implementation
- Answer.AI for developing gpu.cpp and providing a clean C++ API
- The WebGPU community for developing the specification

## Links

- [Documentation](https://hackage.haskell.org/package/webgpu-dawn)
- [Issue Tracker](https://github.com/yourusername/webgpu-dawn/issues)
- [WebGPU Shading Language Spec](https://www.w3.org/TR/WGSL/)

## Contact

Maintainer: Junji Hashimoto <junji.hashimoto@gmail.com>
