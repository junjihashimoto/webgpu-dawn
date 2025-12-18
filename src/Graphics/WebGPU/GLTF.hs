{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

-- | Minimal glTF 2.0 loader that extracts triangle mesh data for use with WebGPU.
--   This loader focuses on static meshes and currently supports POSITION, NORMAL,
--   TEXCOORD_0 attributes, plus unsigned short/unsigned int indices. It is
--   designed for offline asset preparation rather than being a fully featured
--   runtime parser.
module Graphics.WebGPU.GLTF
  ( GLTFError(..)
  , LoadedGLTF(..)
  , LoadedMesh(..)
  , LoadedPrimitive(..)
  , PrimitiveIndices(..)
  , loadGLTF
  ) where

import Control.Exception (IOException, try)
import Control.Monad (replicateM, unless, when)
import Data.Aeson
import Data.Binary.Get (Get, getFloatle, getWord16le, getWord32le, runGetOrFail)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Base64 as B64
import qualified Data.ByteString.Lazy as BL
import qualified Data.HashMap.Strict as HM
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Vector (Vector)
import qualified Data.Vector as V
import qualified Data.Vector.Storable as VS
import Data.Word (Word16, Word32)
import GHC.Generics (Generic)
import System.FilePath (takeDirectory, (</>))

data GLTFError
  = GLTFError String
  deriving (Eq, Show)

data LoadedGLTF = LoadedGLTF
  { gltfMeshes :: [LoadedMesh]
  } deriving (Show)

data LoadedMesh = LoadedMesh
  { meshName :: Maybe Text
  , meshPrimitives :: [LoadedPrimitive]
  } deriving (Show)

data LoadedPrimitive = LoadedPrimitive
  { primitivePositions :: Maybe (VS.Vector Float)
  , primitiveNormals :: Maybe (VS.Vector Float)
  , primitiveTexCoords0 :: Maybe (VS.Vector Float)
  , primitiveIndices :: Maybe PrimitiveIndices
  } deriving (Show)

data PrimitiveIndices
  = IndicesWord16 (VS.Vector Word16)
  | IndicesWord32 (VS.Vector Word32)
  deriving (Show)

-- | Load a glTF file together with its referenced buffers.
--   Returns decoded mesh data for convenience.
loadGLTF :: FilePath -> IO (Either GLTFError LoadedGLTF)
loadGLTF gltfPath = do
  contents <- BL.readFile gltfPath
  case eitherDecode contents of
    Left err -> pure $ Left (GLTFError ("Failed to parse glTF JSON: " ++ err))
    Right rawDoc -> do
      let baseDir = takeDirectory gltfPath
      bufferResults <- mapM (loadBufferEntry baseDir) (zip [0..] (rawBuffers rawDoc))
      case sequence bufferResults of
        Left e -> pure (Left e)
        Right buffers -> do
          let ctx = LoaderContext
                { ctxBuffers = V.fromList buffers
                , ctxBufferViews = V.fromList (rawBufferViews rawDoc)
                , ctxAccessors = V.fromList (rawAccessors rawDoc)
                }
          pure $ LoadedGLTF <$> traverse (loadMesh ctx) (rawMeshes rawDoc)

--------------------------------------------------------------------------------
-- JSON document structures

data RawGLTF = RawGLTF
  { rawBuffers :: [BufferDef]
  , rawBufferViews :: [BufferViewDef]
  , rawAccessors :: [AccessorDef]
  , rawMeshes :: [MeshDef]
  } deriving (Show, Generic)

instance FromJSON RawGLTF where
  parseJSON = withObject "glTF" $ \o -> do
    buffers <- o .:? "buffers" .!= []
    bufferViews <- o .:? "bufferViews" .!= []
    accessors <- o .:? "accessors" .!= []
    meshes <- o .:? "meshes" .!= []
    pure RawGLTF
      { rawBuffers = buffers
      , rawBufferViews = bufferViews
      , rawAccessors = accessors
      , rawMeshes = meshes
      }

data BufferDef = BufferDef
  { bufferUri :: Maybe Text
  , bufferByteLength :: Int
  } deriving (Show, Generic)

instance FromJSON BufferDef where
  parseJSON = withObject "buffer" $ \o ->
    BufferDef <$> o .:? "uri"
              <*> o .: "byteLength"

data BufferViewDef = BufferViewDef
  { bufferViewBuffer :: Int
  , bufferViewByteOffset :: Int
  , bufferViewByteLength :: Int
  , bufferViewByteStride :: Maybe Int
  } deriving (Show, Generic)

instance FromJSON BufferViewDef where
  parseJSON = withObject "bufferView" $ \o ->
    BufferViewDef
      <$> o .: "buffer"
      <*> o .:? "byteOffset" .!= 0
      <*> o .: "byteLength"
      <*> o .:? "byteStride"

data AccessorDef = AccessorDef
  { accessorBufferView :: Maybe Int
  , accessorByteOffset :: Int
  , accessorComponentType :: ComponentType
  , accessorCount :: Int
  , accessorValueType :: AccessorValueType
  , accessorNormalized :: Bool
  } deriving (Show, Generic)

instance FromJSON AccessorDef where
  parseJSON = withObject "accessor" $ \o ->
    AccessorDef
      <$> o .:? "bufferView"
      <*> o .:? "byteOffset" .!= 0
      <*> o .: "componentType"
      <*> o .: "count"
      <*> o .: "type"
      <*> o .:? "normalized" .!= False

data MeshDef = MeshDef
  { meshDefName :: Maybe Text
  , meshDefPrimitives :: [PrimitiveDef]
  } deriving (Show, Generic)

instance FromJSON MeshDef where
  parseJSON = withObject "mesh" $ \o ->
    MeshDef <$> o .:? "name"
            <*> o .:? "primitives" .!= []

data PrimitiveDef = PrimitiveDef
  { primitiveDefAttributes :: HM.HashMap Text Int
  , primitiveDefIndices :: Maybe Int
  } deriving (Show, Generic)

instance FromJSON PrimitiveDef where
  parseJSON = withObject "primitive" $ \o ->
    PrimitiveDef <$> o .:? "attributes" .!= HM.empty
                 <*> o .:? "indices"

data ComponentType
  = ComponentByte
  | ComponentUnsignedByte
  | ComponentShort
  | ComponentUnsignedShort
  | ComponentUnsignedInt
  | ComponentFloat
  deriving (Show, Eq)

instance FromJSON ComponentType where
  parseJSON = withScientific "componentType" $ \num ->
    case truncate num :: Int of
      5120 -> pure ComponentByte
      5121 -> pure ComponentUnsignedByte
      5122 -> pure ComponentShort
      5123 -> pure ComponentUnsignedShort
      5125 -> pure ComponentUnsignedInt
      5126 -> pure ComponentFloat
      v    -> fail $ "Unsupported componentType: " ++ show v

data AccessorValueType
  = ValueScalar
  | ValueVec2
  | ValueVec3
  | ValueVec4
  deriving (Show, Eq)

instance FromJSON AccessorValueType where
  parseJSON = withText "type" $ \t ->
    case t of
      "SCALAR" -> pure ValueScalar
      "VEC2"   -> pure ValueVec2
      "VEC3"   -> pure ValueVec3
      "VEC4"   -> pure ValueVec4
      _        -> fail ("Unsupported accessor type: " ++ T.unpack t)

--------------------------------------------------------------------------------
-- Loader implementation

data LoaderContext = LoaderContext
  { ctxBuffers :: Vector BS.ByteString
  , ctxBufferViews :: Vector BufferViewDef
  , ctxAccessors :: Vector AccessorDef
  }

loadMesh :: LoaderContext -> MeshDef -> Either GLTFError LoadedMesh
loadMesh ctx mesh =
  LoadedMesh (meshDefName mesh)
    <$> traverse (loadPrimitive ctx) (meshDefPrimitives mesh)

loadPrimitive :: LoaderContext -> PrimitiveDef -> Either GLTFError LoadedPrimitive
loadPrimitive ctx prim = do
  let attrs = primitiveDefAttributes prim
      attr name = HM.lookup name attrs
  positions <- traverse (loadFloatVector ctx ValueVec3) (attr "POSITION")
  normals <- traverse (loadFloatVector ctx ValueVec3) (attr "NORMAL")
  tex0 <- traverse (loadFloatVector ctx ValueVec2) (attr "TEXCOORD_0")
  idx <- traverse (loadIndices ctx) (primitiveDefIndices prim)
  pure LoadedPrimitive
    { primitivePositions = positions
    , primitiveNormals = normals
    , primitiveTexCoords0 = tex0
    , primitiveIndices = idx
    }

loadFloatVector :: LoaderContext -> AccessorValueType -> Int -> Either GLTFError (VS.Vector Float)
loadFloatVector ctx expectedType accessorIx = do
  (accessor, bytes) <- accessorBytes ctx accessorIx
  when (accessorValueType accessor /= expectedType) $
    Left $ GLTFError $
      "Accessor " ++ show accessorIx ++ " has incompatible type (expected "
      ++ show expectedType ++ ")"
  unless (accessorComponentType accessor == ComponentFloat) $
    Left $ GLTFError $
      "Accessor " ++ show accessorIx ++ " must use FLOAT componentType"
  when (accessorNormalized accessor) $
    Left $ GLTFError $
      "Accessor " ++ show accessorIx ++ " uses normalized components, which are unsupported"
  decodeFloatVector expectedType accessor bytes

loadIndices :: LoaderContext -> Int -> Either GLTFError PrimitiveIndices
loadIndices ctx accessorIx = do
  (accessor, bytes) <- accessorBytes ctx accessorIx
  when (accessorValueType accessor /= ValueScalar) $
    Left $ GLTFError "Index accessor must be SCALAR type"
  when (accessorNormalized accessor) $
    Left $ GLTFError "Index accessor cannot be normalized"
  case accessorComponentType accessor of
    ComponentUnsignedShort -> IndicesWord16 <$> decodeWord16Vector accessor bytes
    ComponentUnsignedInt -> IndicesWord32 <$> decodeWord32Vector accessor bytes
    other ->
      Left $ GLTFError $ "Unsupported index component type: " ++ show other

accessorBytes :: LoaderContext -> Int -> Either GLTFError (AccessorDef, BS.ByteString)
accessorBytes ctx accessorIx = do
  accessor <- note ("Unknown accessor index: " ++ show accessorIx) $
    ctxAccessors ctx V.!? accessorIx
  viewIx <- note ("Accessor " ++ show accessorIx ++ " is missing bufferView") $
    accessorBufferView accessor
  view <- note ("Unknown bufferView index: " ++ show viewIx) $
    ctxBufferViews ctx V.!? viewIx
  buffer <- note ("Unknown buffer index: " ++ show (bufferViewBuffer view)) $
    ctxBuffers ctx V.!? bufferViewBuffer view
  let elementSize = componentByteSize (accessorComponentType accessor)
                * valueTypeComponents (accessorValueType accessor)
      stride = fromMaybe elementSize (bufferViewByteStride view)
  when (stride /= elementSize) $
    Left $ GLTFError "Interleaved buffer views are not supported"
  let maxAvailable = bufferViewByteLength view - accessorByteOffset accessor
      requiredBytes = accessorCount accessor * elementSize
  when (requiredBytes > maxAvailable) $
    Left $ GLTFError "Accessor data overruns its bufferView"
  slice <- sliceBytes buffer
    (bufferViewByteOffset view + accessorByteOffset accessor)
    requiredBytes
  pure (accessor, slice)

componentByteSize :: ComponentType -> Int
componentByteSize ct =
  case ct of
    ComponentByte -> 1
    ComponentUnsignedByte -> 1
    ComponentShort -> 2
    ComponentUnsignedShort -> 2
    ComponentUnsignedInt -> 4
    ComponentFloat -> 4

valueTypeComponents :: AccessorValueType -> Int
valueTypeComponents vt =
  case vt of
    ValueScalar -> 1
    ValueVec2 -> 2
    ValueVec3 -> 3
    ValueVec4 -> 4

decodeFloatVector :: AccessorValueType -> AccessorDef -> BS.ByteString -> Either GLTFError (VS.Vector Float)
decodeFloatVector valType accessor bytes = do
  let totalValues = accessorCount accessor * valueTypeComponents valType
  floats <- decodeList getFloatle totalValues bytes
  pure (VS.fromList floats)

decodeWord16Vector :: AccessorDef -> BS.ByteString -> Either GLTFError (VS.Vector Word16)
decodeWord16Vector accessor bytes = do
  values <- decodeList getWord16le totalValues bytes
  pure (VS.fromList values)
  where
    totalValues = accessorCount accessor

decodeWord32Vector :: AccessorDef -> BS.ByteString -> Either GLTFError (VS.Vector Word32)
decodeWord32Vector accessor bytes = do
  values <- decodeList getWord32le totalValues bytes
  pure (VS.fromList values)
  where
    totalValues = accessorCount accessor

--------------------------------------------------------------------------------
-- Binary decoding helpers

decodeList :: Get a -> Int -> BS.ByteString -> Either GLTFError [a]
decodeList getter count bytes =
  case runGetOrFail (replicateM count getter) (BL.fromStrict bytes) of
    Left (_, _, err) -> Left $ GLTFError ("Failed to decode accessor payload: " ++ err)
    Right (_, _, values) -> Right values

--------------------------------------------------------------------------------
-- Low-level byte helpers

sliceBytes :: BS.ByteString -> Int -> Int -> Either GLTFError BS.ByteString
sliceBytes bytes offset len
  | offset < 0 || len < 0 = Left $ GLTFError "Negative slice requested"
  | offset + len > BS.length bytes = Left $ GLTFError "Slice exceeds buffer bounds"
  | otherwise = Right $ BS.take len (BS.drop offset bytes)

--------------------------------------------------------------------------------
-- Buffer loading helpers

loadBufferEntry :: FilePath -> (Int, BufferDef) -> IO (Either GLTFError BS.ByteString)
loadBufferEntry baseDir (idx, BufferDef mUri byteLen) =
  case mUri of
    Nothing ->
      pure $ Left $ GLTFError $ "Buffer " ++ show idx ++ " is missing a URI"
    Just uriText
      | "data:" `T.isPrefixOf` uriText ->
          pure $ loadDataUri idx byteLen uriText
      | otherwise -> do
          let filePath = baseDir </> T.unpack uriText
          result <- try (BS.readFile filePath) :: IO (Either IOException BS.ByteString)
          pure $ case result of
            Left err -> Left $ GLTFError $ "Failed to read buffer " ++ show idx ++ ": " ++ show err
            Right bytes -> checkLength idx byteLen bytes

loadDataUri :: Int -> Int -> Text -> Either GLTFError BS.ByteString
loadDataUri idx expectedLen uriText =
  let (_, rest) = T.breakOn "," uriText
  in if T.null rest
       then Left $ GLTFError $ "Buffer " ++ show idx ++ " has malformed data URI"
       else
         let encoded = TE.encodeUtf8 (T.drop 1 rest)
             decoded = B64.decodeLenient encoded
         in checkLength idx expectedLen decoded

checkLength :: Int -> Int -> BS.ByteString -> Either GLTFError BS.ByteString
checkLength idx expected bytes
  | BS.length bytes < expected =
      Left $ GLTFError $
        "Buffer " ++ show idx ++ " shorter than declared byteLength (" ++ show expected ++ ")"
  | BS.length bytes == expected = Right bytes
  | otherwise = Right (BS.take expected bytes)

--------------------------------------------------------------------------------
-- Utility helpers

note :: String -> Maybe a -> Either GLTFError a
note msg = maybe (Left (GLTFError msg)) Right
