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
    Nothing -> return "3f79f3aefe0b0a498002564fcfb13eb21ab6c047"  -- Known working commit
    Just other -> return other

-- Build Dawn using git clone
buildDawnWithGit :: FilePath -> String -> IO ()
buildDawnWithGit dawnSrc dawnVersion = do
  putStrLn "Using git to fetch Dawn repository..."
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

-- Build Dawn by downloading tarball (no git required)
buildDawnWithoutGit :: FilePath -> String -> IO ()
buildDawnWithoutGit dawnSrc dawnVersion = do
  putStrLn "Downloading Dawn source tarball (no git required)..."
  createDirectoryIfMissing True dawnSrc

  -- Download tarball from GitHub mirror or googlesource
  let tarballUrl = "https://dawn.googlesource.com/dawn/+archive/" ++ dawnVersion ++ ".tar.gz"
  putStrLn $ "Downloading from: " ++ tarballUrl

  -- Use temporary file for download
  withSystemTempDirectory "dawn-download" $ \tmpDir -> do
    let tarballPath = tmpDir </> "dawn.tar.gz"

    -- Download tarball
    request <- parseRequest tarballUrl
    response <- httpLBS request
    LBS.writeFile tarballPath (getResponseBody response)

    putStrLn "Extracting tarball..."
    -- Extract tarball to dawnSrc
    callProcess "tar" ["-xzf", tarballPath, "-C", dawnSrc]

  putStrLn $ "Dawn source extracted to: " ++ dawnSrc

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

-- Download pre-built Dawn binaries from GitHub releases
downloadDawnCache :: FilePath -> Bool -> IO Bool
downloadDawnCache dest glfwEnabled = do
  dawnVersion <- getDawnVersion
  let githubRepo = "junjihashimoto/webgpu-dawn"
      -- Use a release tag that matches the Dawn commit
      releaseTag = "v0.1.1"  -- Update this to match your release
      archiveName = "dawn-" ++ dawnVersion ++ "-" ++ platformTag ++ ".tar.gz"
      downloadUrl = "https://github.com/" ++ githubRepo ++ "/releases/download/" ++ releaseTag ++ "/" ++ archiveName
      checksumUrl = downloadUrl ++ ".sha256"

  putStrLn $ "Attempting to download pre-built Dawn from GitHub releases..."
  putStrLn $ "URL: " ++ downloadUrl

  -- Try to download
  result <- try $ do
    createDirectoryIfMissing True dest

    withSystemTempDirectory "dawn-cache-download" $ \tmpDir -> do
      let tarballPath = tmpDir </> archiveName
          checksumPath = tmpDir </> (archiveName ++ ".sha256")

      -- Download archive
      putStrLn "Downloading Dawn binary cache..."
      request <- parseRequest downloadUrl
      response <- httpLBS request
      LBS.writeFile tarballPath (getResponseBody response)

      -- Download and verify checksum
      putStrLn "Downloading checksum..."
      checksumReq <- parseRequest checksumUrl
      checksumResp <- httpLBS checksumReq
      LBS.writeFile checksumPath (getResponseBody checksumResp)

      -- Verify checksum
      putStrLn "Verifying checksum..."
      expectedChecksum <- head . words <$> readFile checksumPath
      (_, actualChecksumOutput, _) <- readProcessWithExitCode "shasum" ["-a", "256", tarballPath] ""
      let actualChecksum = head $ words actualChecksumOutput

      if expectedChecksum /= actualChecksum
        then do
          putStrLn $ "Checksum mismatch! Expected: " ++ expectedChecksum ++ ", Got: " ++ actualChecksum
          return False
        else do
          putStrLn "Checksum verified!"

          -- Extract archive
          putStrLn $ "Extracting to " ++ dest ++ "..."
          callProcess "tar" ["-xzf", tarballPath, "-C", dest]

          putStrLn "✓ Dawn binary cache downloaded successfully!"
          return True

  case result of
    Right success -> return success
    Left (e :: SomeException) -> do
      putStrLn $ "Failed to download cache: " ++ show e
      return False

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
          -- Check if user wants to force building from source
          forceBuild <- lookupEnv "DAWN_BUILD_FROM_SOURCE"
          case forceBuild of
            Just "1" -> do
              -- Force build from source, skip cache download
              putStrLn "DAWN_BUILD_FROM_SOURCE=1 set; building from source..."
              buildDawnTo dest glfwEnabled
              writeFile marker ""
              pure $ Just dest
            _ -> do
              -- Try to download from GitHub releases first (default behavior)
              putStrLn "Dawn not found in local cache."
              downloaded <- downloadDawnCache dest glfwEnabled
              if downloaded
                then do
                  writeFile marker ""
                  pure $ Just dest
                else do
                  -- Fall back to building from source
                  putStrLn "Downloading from cache failed, building from source..."
                  buildDawnTo dest glfwEnabled
                  writeFile marker ""
                  pure $ Just dest

buildDawnTo :: FilePath -> Bool -> IO ()
buildDawnTo dest glfwEnabled = do
  createDirectoryIfMissing True dest

  -- Check if we should use git or direct download
  -- By default, use tarball download (no git required)
  -- Set DAWN_USE_GIT=1 to use git clone instead
  useGit <- lookupEnv "DAWN_USE_GIT"
  dawnVersion <- getDawnVersion
  let dawnSrc = dest </> "src"
      dawnBuild = dest </> "build"

  case useGit of
    Just "1" -> buildDawnWithGit dawnSrc dawnVersion
    _ -> buildDawnWithoutGit dawnSrc dawnVersion

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
