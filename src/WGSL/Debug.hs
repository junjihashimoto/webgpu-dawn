{-# LANGUAGE ForeignFunctionInterface #-}

{-|
Module      : WGSL.Debug
Description : GPU printf debugging with ring buffers
Copyright   : (c) 2025
License     : BSD3

GPU-side printf-style debugging for WebGPU compute kernels.
Enables writing debug values from GPU threads to a ring buffer,
which can be read back and displayed after kernel execution.

= Usage Pattern

1. Create a debug buffer before running kernels
2. Inject debug buffer into shader as a storage buffer
3. Use atomicAdd to append debug entries from GPU
4. After kernel execution, read and decode the buffer
5. Clear buffer for next run

= Example

@
  -- Create debug buffer (64KB)
  debugBuf <- createDebugBuffer ctx 65536

  -- Compile kernel with debug buffer as extra binding
  kernel <- compileKernelWithDebug ctx code tensors debugBuf wgSize

  -- Run kernel
  dispatchKernel ctx kernel

  -- Read debug output
  entries <- readDebugBuffer ctx debugBuf
  mapM_ print entries

  -- Cleanup
  destroyDebugBuffer debugBuf
@
-}

module WGSL.Debug
  ( DebugBuffer
  , DebugEntry(..)
  , createDebugBuffer
  , destroyDebugBuffer
  , readDebugBuffer
  , clearDebugBuffer
  , formatDebugEntry
  ) where

import Foreign
import Foreign.C.Types
import Foreign.C.String (peekCString)
import Data.Word (Word32)
import qualified Graphics.WebGPU.Dawn.Internal as I
import Graphics.WebGPU.Dawn.Types (GPUException(..))
import Control.Exception (throwIO)
import Text.Printf (printf)
import System.IO.Unsafe (unsafePerformIO)

-- | Opaque handle to GPU debug ring buffer
newtype DebugBuffer = DebugBuffer I.DebugBuffer
  deriving (Eq)

-- | A single debug entry from GPU
-- Contains thread ID and up to 8 values
data DebugEntry = DebugEntry
  { entryThreadId :: !Word32
  , entryValues :: ![Float]
  } deriving (Show, Eq)

-- | Create a debug buffer for GPU printf-style debugging
--
-- Buffer size should be large enough to hold debug output from all threads.
-- Recommended: 64KB (65536 bytes) for typical workloads
--
-- Format: [atomic_counter (u32), padding, entries...]
-- Each entry: [thread_id (u32), count (u32), values (f32)...]
createDebugBuffer :: I.Context -> Int -> IO DebugBuffer
createDebugBuffer ctx bufferSize = alloca $ \errPtr -> do
  poke errPtr (I.GPUError 0 nullPtr)

  buf <- I.c_createDebugBuffer ctx (fromIntegral bufferSize) errPtr

  hasErr <- I.c_hasError errPtr
  if hasErr /= 0
    then do
      err <- peek errPtr
      msgPtr <- I.c_getLastErrorMessage errPtr
      msg <- if msgPtr /= nullPtr
             then peekCString msgPtr
             else return "Unknown error"
      throwIO $ GPUError (fromIntegral $ I.errorCode err) msg
    else
      return $ DebugBuffer buf

-- | Destroy debug buffer and free GPU resources
destroyDebugBuffer :: DebugBuffer -> IO ()
destroyDebugBuffer (DebugBuffer buf) = I.c_destroyDebugBuffer buf

-- | Read debug buffer contents after kernel execution
--
-- Returns list of debug entries written by GPU threads.
-- Entries are decoded from raw u32 buffer format.
readDebugBuffer :: I.Context -> DebugBuffer -> IO [DebugEntry]
readDebugBuffer ctx (DebugBuffer buf) = do
  -- Allocate space for maximum entries (adjust as needed)
  let maxWords = 16384  -- 64KB / 4 bytes
  allocaArray maxWords $ \dataPtr -> do
    alloca $ \errPtr -> do
      poke errPtr (I.GPUError 0 nullPtr)

      numRead <- I.c_readDebugBuffer ctx buf dataPtr (fromIntegral maxWords) errPtr

      hasErr <- I.c_hasError errPtr
      if hasErr /= 0
        then do
          err <- peek errPtr
          msgPtr <- I.c_getLastErrorMessage errPtr
          msg <- if msgPtr /= nullPtr
                 then peekCString msgPtr
                 else return "Unknown error"
          throwIO $ GPUError (fromIntegral $ I.errorCode err) msg
        else do
          -- Read raw data
          rawData <- peekArray (fromIntegral numRead) dataPtr

          -- Decode entries
          -- Format: [atomic_counter, padding/entries...]
          -- First word is the atomic counter (number of entries written)
          let entries = decodeDebugEntries rawData
          return entries

-- | Clear debug buffer (reset atomic counter to 0)
-- Call this before each kernel dispatch to reset debug output
clearDebugBuffer :: I.Context -> DebugBuffer -> IO ()
clearDebugBuffer ctx (DebugBuffer buf) = I.c_clearDebugBuffer ctx buf

-- | Decode raw Word32 array into DebugEntry list
-- Format: [counter, entry1_threadId, entry1_count, entry1_val0, ...]
decodeDebugEntries :: [Word32] -> [DebugEntry]
decodeDebugEntries [] = []
decodeDebugEntries (counter:rest) =
  let numEntries = fromIntegral counter
  in take numEntries $ parseEntries rest
  where
    parseEntries :: [Word32] -> [DebugEntry]
    parseEntries [] = []
    parseEntries (threadId:count:vals) =
      let valCount = fromIntegral count
          (entryVals, remaining) = splitAt valCount vals
          floatVals = map word32ToFloat entryVals
          entry = DebugEntry threadId floatVals
      in entry : parseEntries remaining
    parseEntries _ = []  -- Incomplete entry

-- | Convert Word32 bit pattern to Float
word32ToFloat :: Word32 -> Float
word32ToFloat w = unsafePerformIO $
  alloca $ \ptr -> do
    poke ptr w
    peek (castPtr ptr :: Ptr Float)

-- | Format a debug entry for human-readable display
formatDebugEntry :: DebugEntry -> String
formatDebugEntry (DebugEntry threadId vals) =
  printf "[Thread %d] %s" threadId (show vals)
