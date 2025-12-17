module Graphics.WebGPU.Dawn.Context
  ( Context
  , withContext
  , createContext
  , destroyContext
  , checkError
  ) where

import Control.Exception (bracket, throwIO)
import Foreign
import Foreign.C.Types
import Foreign.C.String
import qualified Graphics.WebGPU.Dawn.Internal as I
import Graphics.WebGPU.Dawn.Types

-- | Create a new GPU context with automatic resource management
withContext :: (Context -> IO a) -> IO a
withContext = bracket createContext destroyContext

-- | Create a new GPU context
createContext :: IO Context
createContext = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)
  ctx <- I.c_createContext errPtr
  checkError errPtr
  if ctx == nullPtr
    then throwIO $ GPUError 1 "Failed to create GPU context"
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
