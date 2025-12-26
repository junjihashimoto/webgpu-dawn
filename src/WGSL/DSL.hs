{-# LANGUAGE GADTs #-}
{-# LANGUAGE FlexibleInstances #-}

-- | Top-level WGSL DSL module with operator overloading and builder functions
-- This module consolidates all user-facing DSL functionality
module WGSL.DSL
  ( -- * Re-exports from core modules
    module WGSL.AST
  , module WGSL.Monad
  , module WGSL.CodeGen

    -- * Boolean operators (standalone functions)
  , (&&), (||), not

    -- * Literal constructors and utilities
  , lit
  , litF32
  , litI32
  , litBool
  , (!), (.!)  -- Array indexing
  , vecX, vecY, vecZ  -- Vector accessors
  , fromInteger, fromRational

    -- * Type classes for operator overloading
  , Eq'(..), Ord'(..)

    -- * Builder functions (formerly in WGSL.Builder)
  , computeShader
  , computeShaderWithSize
  , computeShaderWithShared
  , defineFunction
  , runShaderModule
  , buildShaderModule
  , buildShaderModuleWithExtensions
  , buildShaderModuleWithConfig
  , KernelConfig(..)
  , defaultKernelConfig
  , buildKernel
  , buildShaderWithAutoBinding
  ) where

import Prelude hiding ((+), (-), (*), (/), negate, (==), (/=), (<), (<=), (>), (>=), (&&), (||), not, fromInteger, fromRational)
import qualified Prelude as P

import WGSL.AST
import WGSL.Monad
import WGSL.CodeGen

-- ============================================================================
-- Operator Overloading (original DSL functionality)
-- ============================================================================

-- | Numeric operations for F32
instance P.Num (Exp F32) where
  (+) = Add
  (-) = Sub
  (*) = Mul
  negate = Neg
  abs = Abs
  signum x = Var "signum_not_implemented"  -- Placeholder
  fromInteger n = LitF32 (P.fromInteger n)

-- | Numeric operations for I32
instance P.Num (Exp I32) where
  (+) = Add
  (-) = Sub
  (*) = Mul
  negate = Neg
  abs = Abs
  signum x = Var "signum_not_implemented"  -- Placeholder
  fromInteger n = LitI32 (P.fromInteger n)

-- | Fractional for F32
instance P.Fractional (Exp F32) where
  (/) = Div
  fromRational r = LitF32 (P.fromRational r)

-- | Equality and comparison
class Eq' a where
  (==) :: a -> a -> Exp Bool_
  (/=) :: a -> a -> Exp Bool_
  x /= y = Not (x == y)

instance Eq' (Exp F32) where
  (==) = Eq

instance Eq' (Exp I32) where
  (==) = Eq

instance Eq' (Exp U32) where
  (==) = Eq

class Ord' a where
  (<) :: a -> a -> Exp Bool_
  (<=) :: a -> a -> Exp Bool_
  (>) :: a -> a -> Exp Bool_
  (>=) :: a -> a -> Exp Bool_

instance Ord' (Exp F32) where
  (<) = Lt
  (<=) = Le
  (>) = Gt
  (>=) = Ge

instance Ord' (Exp I32) where
  (<) = Lt
  (<=) = Le
  (>) = Gt
  (>=) = Ge

-- | Boolean operators
(&&) :: Exp Bool_ -> Exp Bool_ -> Exp Bool_
(&&) = And

(||) :: Exp Bool_ -> Exp Bool_ -> Exp Bool_
(||) = Or

not :: Exp Bool_ -> Exp Bool_
not = Not

-- | Literal constructors
lit :: a -> Exp a
lit = error "Use specific literal constructors: litF32, litI32, litBool"

litF32 :: Float -> Exp F32
litF32 = LitF32

litI32 :: Int -> Exp I32
litI32 = LitI32

litBool :: Bool -> Exp Bool_
litBool = LitBool

-- | Array indexing operator
(!) :: Exp (Array n a) -> Exp I32 -> Exp a
(!) = Index

-- | Pointer indexing operator
(.!) :: Ptr s (Array n a) -> Exp I32 -> Exp a
(.!) = PtrIndex

-- | Vector accessors
vecX :: Exp (Vec3 a) -> Exp a
vecX = VecX

vecY :: Exp (Vec3 a) -> Exp a
vecY = VecY

vecZ :: Exp (Vec3 a) -> Exp a
vecZ = VecZ

-- | Numeric literals via fromInteger
fromInteger :: Integer -> Exp I32
fromInteger = LitI32 P.. P.fromInteger

fromRational :: Rational -> Exp F32
fromRational = LitF32 P.. P.fromRational

-- ============================================================================
-- Builder Functions (merged from WGSL.Builder)
-- ============================================================================

-- | Create a compute shader function with default workgroup size (256, 1, 1)
computeShader :: String -> ShaderM () -> FunctionDecl
computeShader name body = computeShaderWithSize name (256, 1, 1) body

-- | Create a compute shader function with specified workgroup size
computeShaderWithSize :: String -> (Int, Int, Int) -> ShaderM () -> FunctionDecl
computeShaderWithSize name (x, y, z) body =
  let (_, state) = runShader body
      attrs = [ "compute"
              , "workgroup_size(" ++ show x ++ ", " ++ show y ++ ", " ++ show z ++ ")"
              ]
      -- Add WGSL built-in parameters
      builtinParams =
        [ (Just "global_invocation_id", "global_invocation_id", TVec3 TU32)
        , (Just "local_invocation_id", "local_invocation_id", TVec3 TU32)
        , (Just "workgroup_id", "workgroup_id", TVec3 TU32)
        , (Just "num_workgroups", "num_workgroups", TVec3 TU32)
        ]
  in FunctionDecl
      { funcName = name
      , funcParams = builtinParams
      , funcReturnType = Nothing
      , funcBody = stmts state
      , funcAttributes = attrs
      }

-- | Create a regular function declaration
defineFunction :: String -> [(String, TypeRep)] -> Maybe TypeRep -> ShaderM () -> FunctionDecl
defineFunction name params retType body =
  let (_, state) = runShader body
      -- Convert old-style params (name, type) to new-style (Nothing, name, type)
      params' = map (\(n, t) -> (Nothing, n, t)) params
  in FunctionDecl
      { funcName = name
      , funcParams = params'
      , funcReturnType = retType
      , funcBody = stmts state
      , funcAttributes = []
      }

-- | Create a compute shader WITH shared variables extraction
-- Returns (FunctionDecl, shared variables to add to module globals)
computeShaderWithShared :: String -> (Int, Int, Int) -> ShaderM () -> (FunctionDecl, [(String, TypeRep, MemorySpace)])
computeShaderWithShared name (x, y, z) body =
  let (_, state) = runShader body
      attrs = [ "@compute"
              , "@workgroup_size(" ++ show x ++ ", " ++ show y ++ ", " ++ show z ++ ")"
              ]
      -- Add WGSL built-in parameters
      builtinParams =
        [ (Just "global_invocation_id", "global_invocation_id", TVec3 TU32)
        , (Just "local_invocation_id", "local_invocation_id", TVec3 TU32)
        , (Just "workgroup_id", "workgroup_id", TVec3 TU32)
        , (Just "num_workgroups", "num_workgroups", TVec3 TU32)
        ]
      func = FunctionDecl
        { funcName = name
        , funcParams = builtinParams
        , funcReturnType = Nothing
        , funcBody = stmts state
        , funcAttributes = attrs
        }
      -- Convert shared vars to module globals with Workgroup memory space
      sharedGlobals = map (\(n, t) -> (n, t, MWorkgroup)) (sharedVars state)
  in (func, sharedGlobals)

-- | Build a complete shader module from a compute shader with shared memory
buildShaderModule :: [(String, TypeRep, MemorySpace)]  -- ^ Storage buffer globals
                  -> ShaderM ()                         -- ^ Shader body
                  -> ShaderModule
buildShaderModule storageGlobals body =
  buildShaderModuleWithExtensions storageGlobals [] [] body

-- | Build a complete shader module with extensions (default workgroup size 256x1x1)
buildShaderModuleWithExtensions :: [(String, TypeRep, MemorySpace)]  -- ^ Storage buffer globals
                                -> [String]                           -- ^ Extensions (e.g., "f16")
                                -> [String]                           -- ^ Diagnostics
                                -> ShaderM ()                         -- ^ Shader body
                                -> ShaderModule
buildShaderModuleWithExtensions storageGlobals exts diags body =
  buildShaderModuleWithConfig storageGlobals exts diags (256, 1, 1) body

-- | Build a complete shader module with extensions and custom workgroup size
buildShaderModuleWithConfig :: [(String, TypeRep, MemorySpace)]  -- ^ Storage buffer globals
                            -> [String]                           -- ^ Extensions (e.g., "f16")
                            -> [String]                           -- ^ Diagnostics
                            -> (Int, Int, Int)                    -- ^ Workgroup size
                            -> ShaderM ()                         -- ^ Shader body
                            -> ShaderModule
buildShaderModuleWithConfig storageGlobals exts diags wgSize body =
  let (func, sharedGlobals) = computeShaderWithShared "main" wgSize body
      allGlobals = storageGlobals ++ sharedGlobals
      -- Build binding metadata from storage globals (not shared vars)
      bindings = zip (map (\(name, _, _) -> name) storageGlobals) [0..]
  in ShaderModule
      { moduleFunctions = [func]
      , moduleGlobals = allGlobals
      , moduleStructs = []  -- No user-defined structs by default
      , moduleExtensions = exts
      , moduleDiagnostics = diags
      , moduleBindings = bindings
      }

-- | Create a complete shader module and generate WGSL code
runShaderModule :: [FunctionDecl] -> [(String, TypeRep, MemorySpace)] -> String
runShaderModule functions globals =
  let bindings = zip (map (\(name, _, _) -> name) globals) [0..]
      shaderMod = ShaderModule
        { moduleFunctions = functions
        , moduleGlobals = globals
        , moduleStructs = []  -- No user-defined structs by default
        , moduleExtensions = []
        , moduleDiagnostics = []
        , moduleBindings = bindings
        }
  in generateWGSL shaderMod

-- ============================================================================
-- Kernel Configuration (Pipeline Integration)
-- ============================================================================

-- | Complete kernel configuration including workgroup size and metadata
data KernelConfig = KernelConfig
  { kernelWorkgroupSize :: (Int, Int, Int)
  , kernelExtensions :: [String]
  , kernelDiagnostics :: [String]
  , kernelFunctionName :: String
  , kernelGlobals :: [(String, TypeRep, MemorySpace)]
  }
  deriving (Show, Eq)

-- | Default kernel configuration
-- Workgroup size: 256x1x1, no extensions, function name: "main"
defaultKernelConfig :: KernelConfig
defaultKernelConfig = KernelConfig
  { kernelWorkgroupSize = (256, 1, 1)
  , kernelExtensions = []
  , kernelDiagnostics = []
  , kernelFunctionName = "main"
  , kernelGlobals = []
  }

-- | Build a complete kernel with integrated configuration
-- This is the recommended high-level API for creating shaders
buildKernel :: KernelConfig -> ShaderM () -> ShaderModule
buildKernel config body =
  let (x, y, z) = kernelWorkgroupSize config
      (func, sharedGlobals) = computeShaderWithShared (kernelFunctionName config) (x, y, z) body
      allGlobals = kernelGlobals config ++ sharedGlobals
      bindings = zip (map (\(name, _, _) -> name) (kernelGlobals config)) [0..]
  in ShaderModule
      { moduleFunctions = [func]
      , moduleGlobals = allGlobals
      , moduleStructs = []  -- No user-defined structs by default
      , moduleExtensions = kernelExtensions config
      , moduleDiagnostics = kernelDiagnostics config
      , moduleBindings = bindings
      }

-- ============================================================================
-- Automatic Binding Layout
-- ============================================================================

-- | Build a shader with automatic binding assignment
-- Buffers declared using declareInputBuffer/declareOutputBuffer are automatically
-- assigned binding indices in order of declaration
--
-- This eliminates manual binding management and prevents binding mismatches
buildShaderWithAutoBinding :: (Int, Int, Int)  -- ^ Workgroup size
                           -> ShaderM ()        -- ^ Shader body using declareXxxBuffer
                           -> ShaderModule
buildShaderWithAutoBinding (x, y, z) body =
  let (_, state) = runShader body
      (func, sharedGlobals) = computeShaderWithShared "main" (x, y, z) body
      -- Extract automatically declared buffers from shader state
      autoBuffers = declaredBuffers state
      allGlobals = autoBuffers ++ sharedGlobals
      bindings = zip (map (\(name, _, _) -> name) autoBuffers) [0..]
  in ShaderModule
      { moduleFunctions = [func]
      , moduleGlobals = allGlobals
      , moduleStructs = []
      , moduleExtensions = []
      , moduleDiagnostics = []
      , moduleBindings = bindings
      }
