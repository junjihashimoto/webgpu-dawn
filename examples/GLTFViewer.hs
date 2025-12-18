{-# LANGUAGE CPP #-}

module Main where

#ifdef ENABLE_GLFW

import Control.Monad (unless, when)
import Data.List (intercalate)
import qualified Data.Vector as VB
import qualified Data.Vector.Storable as VS
import Graphics.WebGPU.Dawn.ContT
import Graphics.WebGPU.Dawn.GLFW
import Graphics.WebGPU.GLTF
import Numeric (showFFloat)
import System.Environment (getArgs)

data Vertex = Vertex
  { vPos :: (Float, Float, Float)
  , vNormal :: (Float, Float, Float)
  , vUV :: (Float, Float)
  } deriving (Show)

defaultAsset :: FilePath
defaultAsset = "examples/assets/triangle.gltf"

main :: IO ()
main = do
  args <- getArgs
  let assetPath = case args of
        (path:_) -> path
        []       -> defaultAsset
  putStrLn $ "Loading glTF file: " ++ assetPath
  modelResult <- loadGLTF assetPath
  case modelResult of
    Left err -> putStrLn $ "Failed to load glTF: " ++ show err
    Right doc ->
      case pickFirstPrimitive doc >>= buildVertices of
        Left reason -> putStrLn reason
        Right vertices -> runViewer vertices

runViewer :: [Vertex] -> IO ()
runViewer vertices = do
  when (null vertices) $
    error "Selected primitive produced no vertices."
  let shaderSource = buildShaderSource vertices
      vertexCount = length vertices
  evalContT $ do
    liftIO glfwInit
    window <- createWindow 900 700 "WebGPU glTF Viewer"
    ctx <- createContext
    surface <- createSurfaceForWindow ctx window
    format <- liftIO $ getSurfacePreferredFormat surface
    liftIO $ configureSurface surface 900 700
    shader <- createShaderModule ctx shaderSource
    pipeline <- createRenderPipeline ctx shader format
    liftIO $ do
      putStrLn "Starting render loop..."
      renderLoop ctx window surface pipeline vertexCount
    liftIO glfwTerminate
    liftIO $ putStrLn "Viewer closed."

renderLoop :: Context -> Window -> Surface -> RenderPipeline -> Int -> IO ()
renderLoop ctx window surface pipeline vertexCount = do
  shouldClose <- windowShouldClose window
  unless shouldClose $ do
    evalContT $ do
      texture <- createCurrentTexture surface
      view <- createTextureView texture
      encoder <- createCommandEncoder ctx
      pass <- createRenderPass encoder view
      liftIO $ do
        setRenderPipeline pass pipeline
        draw pass vertexCount
        endRenderPass pass
      commands <- createCommandBuffer encoder
      liftIO $ submitCommand ctx commands
    surfacePresent surface
    pollEvents
    renderLoop ctx window surface pipeline vertexCount

pickFirstPrimitive :: LoadedGLTF -> Either String LoadedPrimitive
pickFirstPrimitive doc =
  case gltfMeshes doc of
    [] -> Left "glTF file does not contain any meshes."
    mesh:_ ->
      case meshPrimitives mesh of
        [] -> Left "The first mesh has no primitives."
        prim:_ -> Right prim

buildVertices :: LoadedPrimitive -> Either String [Vertex]
buildVertices prim = do
  posVec <- maybe (Left "Primitive is missing POSITION data.") Right (primitivePositions prim)
  posList <- toVec3List posVec
  let vertexCount = length posList
      normalsList =
        maybe (Right $ replicate vertexCount (0, 0, 1)) toVec3List (primitiveNormals prim)
      uvList =
        maybe (Right $ replicate vertexCount (0, 0)) toVec2List (primitiveTexCoords0 prim)
  normals <- normalsList
  uvs <- uvList
  let positionsV = VB.fromList posList
      normalsV = VB.fromList normals
      uvsV = VB.fromList uvs
      idxList = case primitiveIndices prim of
        Nothing -> Right [0 .. vertexCount - 1]
        Just inds -> toIndexList inds
  indices <- idxList
  traverse (buildVertex positionsV normalsV uvsV) indices

buildVertex :: VB.Vector (Float, Float, Float)
            -> VB.Vector (Float, Float, Float)
            -> VB.Vector (Float, Float)
            -> Int
            -> Either String Vertex
buildVertex positions normals uvs idx = do
  pos <- indexOrErr positions idx "Position"
  nrm <- pure $ indexWithDefault normals idx (0, 0, 1)
  uv  <- pure $ indexWithDefault uvs idx (0, 0)
  pure $ Vertex pos nrm uv

indexOrErr :: VB.Vector a -> Int -> String -> Either String a
indexOrErr vec idx label =
  maybe (Left $ label ++ " index out of range: " ++ show idx) Right (vec VB.!? idx)

indexWithDefault :: VB.Vector a -> Int -> a -> a
indexWithDefault vec idx def =
  maybe def id (vec VB.!? idx)

toVec3List :: VS.Vector Float -> Either String [(Float, Float, Float)]
toVec3List vec
  | VS.length vec `mod` 3 /= 0 = Left "VEC3 accessor has invalid length."
  | otherwise = Right
      [ (vec VS.! (i * 3), vec VS.! (i * 3 + 1), vec VS.! (i * 3 + 2))
      | i <- [0 .. VS.length vec `div` 3 - 1]
      ]

toVec2List :: VS.Vector Float -> Either String [(Float, Float)]
toVec2List vec
  | VS.length vec `mod` 2 /= 0 = Left "VEC2 accessor has invalid length."
  | otherwise = Right
      [ (vec VS.! (i * 2), vec VS.! (i * 2 + 1))
      | i <- [0 .. VS.length vec `div` 2 - 1]
      ]

toIndexList :: PrimitiveIndices -> Either String [Int]
toIndexList (IndicesWord16 vec) = Right $ map fromIntegral (VS.toList vec)
toIndexList (IndicesWord32 vec) = Right $ map fromIntegral (VS.toList vec)

buildShaderSource :: [Vertex] -> String
buildShaderSource vertices =
  let vertexCount = length vertices
      posLines = intercalate ",\n    " $ map (formatVec3 . vPos) vertices
      normLines = intercalate ",\n    " $ map (formatVec3 . vNormal) vertices
      uvLines = intercalate ",\n    " $ map (formatVec2 . vUV) vertices
  in unlines
      [ "struct VertexOutput {"
      , "  @builtin(position) position : vec4f,"
      , "  @location(0) color : vec3f,"
      , "};"
      , "const POSITIONS : array<vec3f," ++ show vertexCount ++ "> = array<vec3f," ++ show vertexCount ++ ">("
      , "    " ++ posLines
      , ");"
      , "const NORMALS : array<vec3f," ++ show vertexCount ++ "> = array<vec3f," ++ show vertexCount ++ ">("
      , "    " ++ normLines
      , ");"
      , "const UVS : array<vec2f," ++ show vertexCount ++ "> = array<vec2f," ++ show vertexCount ++ ">("
      , "    " ++ uvLines
      , ");"
      , "@vertex"
      , "fn vertexMain(@builtin(vertex_index) idx : u32) -> VertexOutput {"
      , "  var out : VertexOutput;"
      , "  let pos = POSITIONS[idx];"
      , "  let normal = normalize(NORMALS[idx]);"
      , "  let lightDir = normalize(vec3f(0.3, 0.7, 1.0));"
      , "  let diffuse = max(dot(normal, lightDir), 0.0);"
      , "  out.position = vec4f(pos, 1.0);"
      , "  out.color = vec3f(0.2, 0.45, 0.8) * diffuse + vec3f(0.05, 0.05, 0.05);"
      , "  return out;"
      , "}"
      , "@fragment"
      , "fn fragmentMain(input : VertexOutput) -> @location(0) vec4f {"
      , "  return vec4f(input.color, 1.0);"
      , "}"
      ]

formatVec3 :: (Float, Float, Float) -> String
formatVec3 (x, y, z) =
  "vec3f(" ++ fmt x ++ ", " ++ fmt y ++ ", " ++ fmt z ++ ")"

formatVec2 :: (Float, Float) -> String
formatVec2 (u, v) =
  "vec2f(" ++ fmt u ++ ", " ++ fmt v ++ ")"

fmt :: Float -> String
fmt val =
  let rendered = showFFloat (Just 4) val ""
  in if '.' `elem` rendered then rendered else rendered ++ ".0"

#else

main :: IO ()
main = putStrLn "This example requires GLFW support. Build with -fglfw."

#endif
