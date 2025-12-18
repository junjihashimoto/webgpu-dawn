module Graphics.WebGPU.Dawn.Context
  ( Context
  , withContext
  , withContextFeatures
  , createContext
  , createContextWithFeatures
  , destroyContext
  , checkError
  -- * WebGPU Feature Names
  , WGPUFeatureName(..)
  ) where

import Control.Exception (bracket, throwIO)
import Foreign
import Foreign.C.Types
import Foreign.C.String
import Data.Word (Word32)
import qualified Graphics.WebGPU.Dawn.Internal as I
import Graphics.WebGPU.Dawn.Types

-- | WebGPU Feature Names (from webgpu.h)
-- These correspond to WGPUFeatureName enum values from Dawn
data WGPUFeatureName
  = FeatureShaderF16                        -- 0x0000000B (11)
  | FeatureSubgroups                        -- 0x00000012 (18)
  | FeatureChromiumExperimentalSubgroupMatrix  -- 0x00050037 (327735)
  deriving (Eq, Show)

-- Convert feature to Word32 value
featureToWord32 :: WGPUFeatureName -> Word32
featureToWord32 FeatureShaderF16 = 0x0000000B
featureToWord32 FeatureSubgroups = 0x00000012
featureToWord32 FeatureChromiumExperimentalSubgroupMatrix = 0x00050037

-- | Create a new GPU context with automatic resource management
withContext :: (Context -> IO a) -> IO a
withContext = bracket createContext destroyContext

-- | Create a new GPU context with features
withContextFeatures :: [String] -> [WGPUFeatureName] -> (Context -> IO a) -> IO a
withContextFeatures toggles features =
  bracket (createContextWithFeatures toggles features) destroyContext

-- | Create a new GPU context
createContext :: IO Context
createContext = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  ctx <- I.c_createContext errPtr
  checkError errPtr
  if ctx == nullPtr
    then throwIO $ GPUError 1 "Failed to create GPU context"
    else return $ Context ctx

-- | Create context with specific device features and toggles
createContextWithFeatures :: [String] -> [WGPUFeatureName] -> IO Context
createContextWithFeatures toggles features = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)

  -- Convert features to Word32 array
  let featureWords = map featureToWord32 features

  -- Create toggles as C strings
  withMany withCString toggles $ \togglePtrs ->
    withArray togglePtrs $ \toggleArr ->
      withArray featureWords $ \featureArr -> do
        ctx <- I.c_createContextWithFeatures
          toggleArr (fromIntegral $ length toggles)
          featureArr (fromIntegral $ length features)
          errPtr
        checkError errPtr
        if ctx == nullPtr
          then throwIO $ GPUError 1 "Failed to create GPU context with features"
          else return $ Context ctx

-- | Destroy a GPU context
destroyContext :: Context -> IO ()
destroyContext (Context ctx) = I.c_destroyContext ctx

-- Internal helper to check for errors
checkError :: Ptr I.GPUError -> IO ()
checkError errPtr = do
  hasErr <- I.c_hasError errPtr
  if hasErr /= 0
    then do
      err <- peek errPtr
      msgPtr <- I.c_getLastErrorMessage errPtr
      if msgPtr /= nullPtr
        then do
          msg <- peekCString msgPtr
          throwIO $ GPUError (fromIntegral $ I.errorCode err) msg
        else
          throwIO $ GPUError (fromIntegral $ I.errorCode err) "Unknown error"
    else
      return ()
