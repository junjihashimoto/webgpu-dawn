# Graphics Examples (GLFW)

This directory contains examples that use **GLFW** for windowed graphics applications.

## Requirements

These examples require the `glfw` flag to be enabled when building:

```bash
cabal build -f glfw
```

## What is GLFW?

GLFW is a cross-platform library for creating windows with OpenGL/WebGPU contexts. These examples demonstrate rendering graphics to a window using WebGPU via the Dawn implementation.

## Examples

### HelloGPU.hs
Basic "Hello World" for GPU graphics. Opens a window and clears it with a color.

### TriangleGUI.hs
Classic "Hello Triangle" example. Renders a colored triangle to the screen.
Demonstrates:
- Vertex buffers
- Render pipelines
- Basic rasterization

### TetrisGUI.hs
Interactive Tetris game implementation using WebGPU for rendering.
Shows more complex graphics programming:
- Game state management
- User input handling
- 2D sprite rendering

### GLTFViewer.hs
GLTF (GL Transmission Format) 3D model viewer.
Demonstrates:
- Loading 3D models
- Texture mapping
- 3D transformations
- Camera controls

## How to Run

```bash
# Make sure glfw flag is enabled
cabal run -f glfw triangle-gui
cabal run -f glfw tetris-gui
cabal run -f glfw gltf-viewer
```

## Graphics vs Compute

| Graphics (this directory) | Compute (`examples/compute/`, `examples/dsl/`) |
|--------------------------|------------------------------------------------|
| Render to screen         | Calculate results in buffers |
| Uses render pipelines    | Uses compute pipelines |
| Vertex/fragment shaders  | Compute shaders |
| Visual output            | Numerical output |
| GLFW windowing           | Headless execution |

## Platform Support

Graphics examples require platform-specific windowing support:
- **macOS**: Uses Metal backend + Cocoa framework
- **Linux**: Requires X11/Wayland
- **Windows**: Uses DirectX 12 backend

The compute examples (`examples/compute/` and `examples/dsl/`) work without windowing and are more portable.
