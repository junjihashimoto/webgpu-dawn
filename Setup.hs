{-# LANGUAGE CPP #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

import Data.List (isPrefixOf, isInfixOf)
import Distribution.Simple
import Distribution.Simple.Program
import Distribution.Simple.Setup
import Distribution.Simple.LocalBuildInfo
import Distribution.Types.LocalBuildInfo
import Distribution.Types.GenericPackageDescription
import Distribution.Types.HookedBuildInfo
import Distribution.Types.BuildInfo
import Distribution.PackageDescription
import Distribution.System
import Distribution.Types.Flag
import Distribution.Types.PackageDescription
import System.Directory
import System.FilePath
import System.Process
import System.Exit (ExitCode(..))
import System.IO.Temp
import Control.Monad
import Network.HTTP.Simple
import qualified Data.ByteString.Lazy as LBS
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BS8
import System.Environment (lookupEnv)
import Data.Maybe (fromMaybe)
import Control.Exception

#if MIN_VERSION_Cabal(3,14,0)
import Distribution.Utils.Path (makeSymbolicPath)
#else
makeSymbolicPath :: a -> a
makeSymbolicPath = id
#endif

main :: IO ()
main = defaultMainWithHooks $ simpleUserHooks
  { preConf = \_ _ -> do
      pure emptyHookedBuildInfo
  , confHook = \(gpd, hbi) flags -> do
      -- Check if GLFW flag is enabled
      let glfwEnabled = lookupFlagAssignment (mkFlagName "glfw") (configConfigurationsFlags flags) == Just True

      mDawnDir <- ensureDawn glfwEnabled
      case mDawnDir of
        Nothing -> do
          putStrLn "Dawn not found or handled by Nix, skipping configuration."
          lbi <- confHook simpleUserHooks (gpd, hbi) flags
          -- For macOS, add the -ld_classic flag to the linker
          case buildOS of
            OSX -> return $ lbi { withPrograms = addLdClassicFlag (withPrograms lbi) }
            _ -> return lbi
        Just dawnDir -> do
          dawnDir <- getLocalUserDawnDir

          -- gpu.hpp is now included in cbits/ directory
          -- No need to clone gpu.cpp repository

          let libDir     = dawnDir </> "lib"
              includeDir = dawnDir </> "include"
              genIncludeDir = dawnDir </> "gen" </> "include"
              -- Also add source include directories for webgpu.h and GLFW
              srcIncludeDir = dawnDir </> "src" </> "include"
              srcWebGPUIncludeDir = srcIncludeDir </> "webgpu"
              glfwIncludeDir = dawnDir </> "src" </> "third_party" </> "glfw" </> "include"

          let updatedFlags = flags
                { configExtraLibDirs      = makeSymbolicPath libDir : configExtraLibDirs flags
                , configExtraIncludeDirs  =
                    makeSymbolicPath srcIncludeDir
                    : makeSymbolicPath includeDir
                    : makeSymbolicPath genIncludeDir
                    : makeSymbolicPath (genIncludeDir </> "dawn")
                    : configExtraIncludeDirs flags
                }
          -- Update the package description to add include paths to BuildInfo
          -- Note: gpu.hpp is in cbits/ directory, GLFW headers from Dawn
          -- Note: GLFW and dawn_glfw are built as separate static libraries when DAWN_USE_GLFW=ON
          let glfwIncludeDirs = if glfwEnabled then [glfwIncludeDir] else []
              glfwLibDir = dawnDir </> "build" </> "third_party" </> "glfw" </> "src"
              dawnGlfwLibDir = dawnDir </> "build" </> "src" </> "dawn" </> "glfw"
              glfwLibs = if glfwEnabled then ["dawn_glfw", "glfw3"] else []
              glfwLibDirs = if glfwEnabled then [dawnGlfwLibDir, glfwLibDir] else []
              updateBuildInfo bi = bi
                { includeDirs = srcWebGPUIncludeDir : srcIncludeDir : (glfwIncludeDirs ++ includeDirs bi)
                , ccOptions = ("-I" ++ srcWebGPUIncludeDir) : (map ("-I" ++) glfwIncludeDirs ++ ccOptions bi)
                , cxxOptions = ("-I" ++ srcWebGPUIncludeDir) : (map ("-I" ++) glfwIncludeDirs ++ cxxOptions bi)
                , extraLibDirs = libDir : (glfwLibDirs ++ extraLibDirs bi)
                , extraLibs = "webgpu_dawn" : (glfwLibs ++ extraLibs bi)
                }
              updateLib lib = lib { libBuildInfo = updateBuildInfo (libBuildInfo lib) }
              gpd' = gpd { condLibrary = fmap updateLib <$> condLibrary gpd }

          -- Call the default configuration hook with updated flags and package description
          lbi <- confHook simpleUserHooks (gpd', hbi) updatedFlags

          -- Add RPath so the binary finds the libs at runtime
          case buildOS of
            OSX -> return $ lbi { withPrograms = addRPath libDir $ addLdClassicFlag (withPrograms lbi) }
            Linux -> return $ lbi { withPrograms = addRPath libDir (withPrograms lbi) }
            _ -> return lbi
  }

getDawnVersion :: IO String
getDawnVersion = do
  mVersion <- lookupEnv "DAWN_VERSION"
  case mVersion of
    Nothing -> return "e1d6e12337080cf9f6d8726209e86df449bc6e9a"  -- Known working commit
    Just other -> return other

getGpuCppVersion :: IO String
getGpuCppVersion = do
  mVersion <- lookupEnv "GPU_CPP_VERSION"
  case mVersion of
    Nothing -> return "dev"  -- Use dev branch
    Just other -> return other

getLocalUserDawnDir :: IO FilePath
getLocalUserDawnDir = do
  mHome <- lookupEnv "DAWN_HOME"
  dawnVersion <- getDawnVersion
  base <- case mHome of
    Just h  -> pure h
    Nothing -> do
      -- XDG cache (Linux/macOS). Falls back to ~/.cache
      cache <- getXdgDirectory XdgCache "dawn"
      pure cache
  pure $ base </> dawnVersion </> platformTag

platformTag :: FilePath
platformTag =
  case (buildOS, buildArch) of
    (OSX,    AArch64) -> "macos-arm64"
    (OSX,    X86_64)  -> "macos-x86_64"
    (Linux,  X86_64)  -> "linux-x86_64"
    -- add more as needed
    _ -> error $ "Unsupported platform: " <> show (buildOS, buildArch)

ensureDawn :: Bool -> IO (Maybe FilePath)
ensureDawn glfwEnabled = do
  isSandbox <- isNixSandbox
  if isSandbox
    then return Nothing
    else buildDawn glfwEnabled

isNixSandbox :: IO Bool
isNixSandbox = do
  nix <- lookupEnv "NIX_BUILD_TOP"
  case nix of
    Just path -> do
      let isNixPath = any (`isPrefixOf` path) ["/build", "/private/tmp/nix-build"]
      if isNixPath
        then do
          putStrLn "Nix sandbox detected; skipping Dawn build."
          return True
        else do
          return False
    Nothing -> return False

buildDawn :: Bool -> IO (Maybe FilePath)
buildDawn glfwEnabled = do
  skip <- lookupEnv "DAWN_SKIP_BUILD"
  case skip of
    Just _ -> do
      putStrLn "DAWN_SKIP_BUILD set; assuming Dawn exists globally."
      return Nothing
    Nothing -> do
      dest <- getLocalUserDawnDir
      let marker = dest </> ".ok"
      exists <- doesFileExist marker
      present <- doesDirectoryExist dest
      if present && exists
        then pure $ Just dest
        else do
          putStrLn $ "Dawn not found in local cache, building to " <> dest
          buildDawnTo dest glfwEnabled

          -- Create an idempotence marker
          writeFile marker ""
          pure $ Just dest

buildDawnTo :: FilePath -> Bool -> IO ()
buildDawnTo dest glfwEnabled = do
  createDirectoryIfMissing True dest

  -- Clone or update Dawn repository
  dawnVersion <- getDawnVersion
  let dawnSrc = dest </> "src"
      dawnBuild = dest </> "build"

  putStrLn "Cloning Dawn repository..."
  createDirectoryIfMissing True dawnSrc

  -- Initialize git repo
  callProcess "git" ["init", dawnSrc]

  -- Check if origin exists
  (_, _, _, ph) <- createProcess (proc "git" ["remote", "get-url", "origin"])
    { cwd = Just dawnSrc
    , std_out = CreatePipe
    , std_err = CreatePipe
    }
  exitCode <- waitForProcess ph

  case exitCode of
    ExitSuccess ->
      callProcess "git" ["-C", dawnSrc, "remote", "set-url", "origin", "https://dawn.googlesource.com/dawn"]
    ExitFailure _ ->
      callProcess "git" ["-C", dawnSrc, "remote", "add", "origin", "https://dawn.googlesource.com/dawn"]

  -- Fetch and checkout specific commit
  putStrLn $ "Fetching Dawn commit: " ++ dawnVersion
  callProcess "git" ["-C", dawnSrc, "fetch", "origin", dawnVersion]
  callProcess "git" ["-C", dawnSrc, "checkout", dawnVersion]
  callProcess "git" ["-C", dawnSrc, "reset", "--hard", dawnVersion]

  -- Apply macOS fix if needed (using ByteString for strict IO)
  when (buildOS == OSX) $ do
    let portDefaultFile = dawnSrc </> "src" </> "dawn" </> "native" </> "metal" </> "PhysicalDeviceMTL.mm"
    portDefaultExists <- doesFileExist portDefaultFile
    when portDefaultExists $ do
      contentBS <- BS.readFile portDefaultFile
      let content = BS8.unpack contentBS
          fixed = replaceAll content "kIOMainPortDefault" "0"
      BS.writeFile portDefaultFile (BS8.pack fixed)

  -- Configure with CMake
  putStrLn $ "Configuring Dawn with CMake (GLFW: " ++ (if glfwEnabled then "enabled" else "disabled") ++ ")..."
  createDirectoryIfMissing True dawnBuild

  let glfwFlag = if glfwEnabled then "ON" else "OFF"
      cmakeArgs =
        [ "-S", dawnSrc
        , "-B", dawnBuild
        , "-DCMAKE_BUILD_TYPE=Release"
        , "-DDAWN_FETCH_DEPENDENCIES=ON"
        , "-DDAWN_BUILD_SAMPLES=OFF"
        , "-DDAWN_BUILD_EXAMPLES=OFF"
        , "-DDAWN_BUILD_TESTS=OFF"
        , "-DDAWN_ENABLE_INSTALL=ON"
        , "-DDAWN_BUILD_MONOLITHIC_LIBRARY=STATIC"
        , "-DCMAKE_INSTALL_PREFIX=" ++ dest
        , "-DTINT_BUILD_TESTS=OFF"
        , "-DTINT_BUILD_IR_BINARY=OFF"
        , "-DTINT_BUILD_CMD_TOOLS=OFF"
        , "-DDAWN_ENABLE_GLFW=" ++ glfwFlag
        , "-DDAWN_USE_GLFW=" ++ glfwFlag
        ]

  callProcess "cmake" cmakeArgs

  -- Build Dawn
  putStrLn "Building Dawn (this may take 10-15 minutes on first build)..."
  callProcess "cmake" ["--build", dawnBuild, "--config", "Release", "--parallel"]

  -- Install Dawn to dest
  putStrLn "Installing Dawn..."
  callProcess "cmake" ["--install", dawnBuild, "--config", "Release"]

  putStrLn "Dawn built successfully."

-- GPU.cpp setup functions
getLocalUserGpuCppDir :: IO FilePath
getLocalUserGpuCppDir = do
  mHome <- lookupEnv "GPU_CPP_HOME"
  gpuCppVersion <- getGpuCppVersion
  base <- case mHome of
    Just h  -> pure h
    Nothing -> do
      cache <- getXdgDirectory XdgCache "gpu-cpp"
      pure cache
  pure $ base </> gpuCppVersion

ensureGpuCpp :: IO FilePath
ensureGpuCpp = do
  skip <- lookupEnv "GPU_CPP_SKIP_DOWNLOAD"
  case skip of
    Just _ -> do
      putStrLn "GPU_CPP_SKIP_DOWNLOAD set; assuming gpu.cpp exists globally."
      return "."
    Nothing -> do
      dest <- getLocalUserGpuCppDir
      let marker = dest </> ".ok"
      exists <- doesFileExist marker
      present <- doesDirectoryExist dest
      if present && exists
        then pure dest
        else do
          putStrLn $ "Cloning gpu.cpp to " <> dest
          cloneGpuCpp dest
          writeFile marker ""
          pure dest

cloneGpuCpp :: FilePath -> IO ()
cloneGpuCpp dest = do
  createDirectoryIfMissing True dest

  gpuCppVersion <- getGpuCppVersion

  putStrLn "Cloning gpu.cpp repository..."

  -- Initialize git repo
  callProcess "git" ["init", dest]

  -- Check if origin exists
  (_, _, _, ph) <- createProcess (proc "git" ["remote", "get-url", "origin"])
    { cwd = Just dest
    , std_out = CreatePipe
    , std_err = CreatePipe
    }
  exitCode <- waitForProcess ph

  case exitCode of
    ExitSuccess ->
      callProcess "git" ["-C", dest, "remote", "set-url", "origin", "https://github.com/AnswerDotAI/gpu.cpp"]
    ExitFailure _ ->
      callProcess "git" ["-C", dest, "remote", "add", "origin", "https://github.com/AnswerDotAI/gpu.cpp"]

  -- Fetch and checkout
  putStrLn $ "Fetching gpu.cpp version: " ++ gpuCppVersion
  callProcess "git" ["-C", dest, "fetch", "origin", gpuCppVersion]
  callProcess "git" ["-C", dest, "checkout", gpuCppVersion]
  callProcess "git" ["-C", dest, "reset", "--hard", "origin/" ++ gpuCppVersion]

  putStrLn "gpu.cpp cloned successfully."

replaceAll :: String -> String -> String -> String
replaceAll str from to =
  let parts = splitOn from str
  in concat $ insertBetween to parts
  where
    splitOn :: String -> String -> [String]
    splitOn needle haystack =
      case breakOn needle haystack of
        (before, match) ->
          if null match
            then [before]
            else before : splitOn needle (drop (length needle) match)

    breakOn :: String -> String -> (String, String)
    breakOn needle haystack =
      case findIndex (isPrefixOf needle) (tails haystack) of
        Nothing -> (haystack, "")
        Just i -> splitAt i haystack

    findIndex :: (a -> Bool) -> [a] -> Maybe Int
    findIndex pred xs = lookup True (zip (map pred xs) [0..])

    tails :: [a] -> [[a]]
    tails [] = [[]]
    tails xs@(_:xs') = xs : tails xs'

    insertBetween :: a -> [a] -> [a]
    insertBetween _ [] = []
    insertBetween _ [x] = [x]
    insertBetween sep (x:xs) = x : sep : insertBetween sep xs

-- Add -ld_classic flag to GHC program arguments for macOS
addLdClassicFlag :: ProgramDb -> ProgramDb
addLdClassicFlag progDb =
  case lookupProgram ghcProgram progDb of
    Just ghc ->
      let ghc' = ghc { programOverrideArgs = ["-optl-ld_classic"] ++ programOverrideArgs ghc }
      in updateProgram ghc' progDb
    Nothing -> progDb

addRPath :: FilePath -> ProgramDb -> ProgramDb
addRPath libDir progDb =
  userSpecifyArgs (programName ldProgram)
  ["-Wl,-rpath," ++ libDir]
  progDb
