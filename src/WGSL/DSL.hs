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
  , litU32
  , litF16
  , litBool
  , (!), (.!)  -- Array indexing
  , vecX, vecY, vecZ  -- Vector accessors
  , (^.)  -- Struct field access
  , fromInteger, fromRational

    -- * Math functions
  , sqrt', abs', min', max', exp', cos', sin', pow', tanh', clamp'

    -- * Type-cast helpers
  , i32, u32, f32, f16

    -- * Bitwise operations (for U32)
  , shiftLeft, shiftRight, bitAnd, bitOr, bitXor
  , (.<<.), (.>>.), (.&.), (.|.), (.^.)  -- Infix operators

    -- * Integer operations
  , divExp, modExp
  , (./.), (.%)

    -- * Type classes for operator overloading
  , Eq'(..), Ord'(..)

    -- * Texture operations
  , textureSample
  , textureLoad
  , textureStore_

    -- * Atomic operations
  , atomicAdd, atomicAddU
  , atomicSub, atomicSubU
  , atomicMin, atomicMinU
  , atomicMax, atomicMaxU
  , atomicExchange, atomicExchangeU
  , atomicCompareExchangeWeak, atomicCompareExchangeWeakU

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

-- | Numeric operations for U32
instance P.Num (Exp U32) where
  (+) = Add
  (-) = Sub
  (*) = Mul
  negate x = Var "negate_not_supported_for_u32"  -- U32 is unsigned
  abs x = x  -- U32 is always non-negative
  signum x = Var "signum_not_implemented"  -- Placeholder
  fromInteger n = LitU32 (P.fromInteger n)

-- | Numeric operations for F16
instance P.Num (Exp F16) where
  (+) = Add
  (-) = Sub
  (*) = Mul
  negate = Neg
  abs = Abs
  signum x = Var "signum_not_implemented"  -- Placeholder
  fromInteger n = LitF16 (P.fromInteger n)

-- | Fractional for F32
instance P.Fractional (Exp F32) where
  (/) = Div
  fromRational r = LitF32 (P.fromRational r)

-- | Fractional for F16
instance P.Fractional (Exp F16) where
  (/) = Div
  fromRational r = LitF16 (P.fromRational r)

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

instance Eq' (Exp F16) where
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

instance Ord' (Exp U32) where
  (<) = Lt
  (<=) = Le
  (>) = Gt
  (>=) = Ge

instance Ord' (Exp F16) where
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

litU32 :: Int -> Exp U32
litU32 n = LitU32 (P.fromIntegral n)

litF16 :: Float -> Exp F16
litF16 = LitF16

litBool :: Bool -> Exp Bool_
litBool = LitBool

-- | Type-cast helper functions for natural syntax
-- Note: These work on literal values. For runtime conversion, use explicit constructors.
i32 :: Exp U32 -> Exp I32
i32 (LitU32 n) = LitI32 (P.fromIntegral n)
i32 e = U32ToI32 e  -- For runtime expressions

u32 :: Exp I32 -> Exp U32
u32 (LitI32 n) = LitU32 (P.fromIntegral n)
u32 e = I32ToU32 e  -- For runtime expressions

-- | Convert literal integers to F32
f32 :: Int -> Exp F32
f32 = LitF32 P.. P.fromIntegral

-- | Convert literal integers to F16
f16 :: Int -> Exp F16
f16 = LitF16 P.. P.fromIntegral

-- ============================================================================
-- Bitwise Operations
-- ============================================================================

-- | Bitwise shift left (<<)
shiftLeft :: Exp U32 -> Exp U32 -> Exp U32
shiftLeft = ShiftLeft

-- | Bitwise shift right (>>)
shiftRight :: Exp U32 -> Exp U32 -> Exp U32
shiftRight = ShiftRight

-- | Bitwise AND (&)
bitAnd :: Exp U32 -> Exp U32 -> Exp U32
bitAnd = BitAnd

-- | Bitwise OR (|)
bitOr :: Exp U32 -> Exp U32 -> Exp U32
bitOr = BitOr

-- | Bitwise XOR (^)
bitXor :: Exp U32 -> Exp U32 -> Exp U32
bitXor = BitXor

-- | Infix operator for shift left
(.<<.) :: Exp U32 -> Exp U32 -> Exp U32
(.<<.) = ShiftLeft
infixl 8 .<<.

-- | Infix operator for shift right
(.>>.) :: Exp U32 -> Exp U32 -> Exp U32
(.>>.) = ShiftRight
infixl 8 .>>.

-- | Infix operator for bitwise AND
(.&.) :: Exp U32 -> Exp U32 -> Exp U32
(.&.) = BitAnd
infixl 7 .&.

-- | Infix operator for bitwise OR
(.|.) :: Exp U32 -> Exp U32 -> Exp U32
(.|.) = BitOr
infixl 5 .|.

-- | Infix operator for bitwise XOR
(.^.) :: Exp U32 -> Exp U32 -> Exp U32
(.^.) = BitXor
infixl 6 .^.

-- ============================================================================
-- Integer Operations
-- ============================================================================

-- | Integer division
divExp :: Exp a -> Exp a -> Exp a
divExp = Div

-- | Modulo operation (remainder)
modExp :: Exp a -> Exp a -> Exp a
modExp = Mod

-- | Infix operator for division
(./.) :: Exp a -> Exp a -> Exp a
(./.) = Div
infixl 7 ./.

-- | Infix operator for modulo
(.%) :: Exp a -> Exp a -> Exp a
(.%) = Mod
infixl 7 .%

-- ============================================================================
-- Array and Vector Operations
-- ============================================================================

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

-- | Struct field access operator
-- Usage: structExpr ^. "fieldName"
(^.) :: Exp (Struct s) -> String -> Exp a
(^.) = FieldAccess

-- | Numeric literals via fromInteger
fromInteger :: Integer -> Exp I32
fromInteger = LitI32 P.. P.fromInteger

fromRational :: Rational -> Exp F32
fromRational = LitF32 P.. P.fromRational

-- ============================================================================
-- Texture Operations
-- ============================================================================

-- | Sample a texture using a sampler at normalized UV coordinates (0.0-1.0)
-- Returns a vec4<f32> containing RGBA values
--
-- Example:
-- @
--   color <- textureSample myTexture mySampler (Vec2 u v)
-- @
textureSample :: Exp (Texture2D format) -> Exp Sampler -> Exp (Vec2 F32) -> Exp (Vec4 F32)
textureSample = TextureSample

-- | Load raw texel data at pixel coordinates and mip level
-- Returns a vec4<f32> containing RGBA values
--
-- Example:
-- @
--   texel <- textureLoad myTexture (Vec2 x y) 0  -- mip level 0
-- @
textureLoad :: Exp (Texture2D format) -> Exp (Vec2 I32) -> Exp I32 -> Exp (Vec4 F32)
textureLoad = TextureLoad

-- | Store texel data at pixel coordinates (for storage textures)
-- This is a monadic operation that generates a statement
--
-- Example:
-- @
--   textureStore_ myTexture (Vec2 x y) (Vec4 r g b a)
-- @
textureStore_ :: Exp (Texture2D format) -> Exp (Vec2 I32) -> Exp (Vec4 F32) -> ShaderM ()
textureStore_ texture coords value = emitStmt $ TextureStore texture coords value

-- ============================================================================
-- Atomic Operations
-- ============================================================================

-- | Atomically add a value to a location, returns the OLD value before addition
--
-- Example:
-- @
--   oldValue <- atomicAdd counterPtr 1  -- Increment counter
-- @
atomicAdd :: Ptr space AtomicI32 -> Exp I32 -> ShaderM (Exp I32)
atomicAdd ptr value = return $ AtomicAdd ptr value

atomicAddU :: Ptr space AtomicU32 -> Exp U32 -> ShaderM (Exp U32)
atomicAddU ptr value = return $ AtomicAddU ptr value

-- | Atomically subtract a value from a location, returns the OLD value
atomicSub :: Ptr space AtomicI32 -> Exp I32 -> ShaderM (Exp I32)
atomicSub ptr value = return $ AtomicSub ptr value

atomicSubU :: Ptr space AtomicU32 -> Exp U32 -> ShaderM (Exp U32)
atomicSubU ptr value = return $ AtomicSubU ptr value

-- | Atomically compute minimum, returns the OLD value
--
-- Example:
-- @
--   oldMin <- atomicMin minPtr newValue  -- Track global minimum
-- @
atomicMin :: Ptr space AtomicI32 -> Exp I32 -> ShaderM (Exp I32)
atomicMin ptr value = return $ AtomicMin ptr value

atomicMinU :: Ptr space AtomicU32 -> Exp U32 -> ShaderM (Exp U32)
atomicMinU ptr value = return $ AtomicMinU ptr value

-- | Atomically compute maximum, returns the OLD value
atomicMax :: Ptr space AtomicI32 -> Exp I32 -> ShaderM (Exp I32)
atomicMax ptr value = return $ AtomicMax ptr value

atomicMaxU :: Ptr space AtomicU32 -> Exp U32 -> ShaderM (Exp U32)
atomicMaxU ptr value = return $ AtomicMaxU ptr value

-- | Atomically exchange (swap) values, returns the OLD value
--
-- Example:
-- @
--   oldFlag <- atomicExchange flagPtr 1  -- Set flag, get old value
-- @
atomicExchange :: Ptr space AtomicI32 -> Exp I32 -> ShaderM (Exp I32)
atomicExchange ptr value = return $ AtomicExchange ptr value

atomicExchangeU :: Ptr space AtomicU32 -> Exp U32 -> ShaderM (Exp U32)
atomicExchangeU ptr value = return $ AtomicExchangeU ptr value

-- | Atomically compare and exchange weak
-- Compares value at ptr with comparand, and if equal, replaces with value
-- Returns the OLD value before the operation
--
-- Example:
-- @
--   oldValue <- atomicCompareExchangeWeak lockPtr 0 1  -- Try to acquire lock
-- @
atomicCompareExchangeWeak :: Ptr space AtomicI32 -> Exp I32 -> Exp I32 -> ShaderM (Exp I32)
atomicCompareExchangeWeak ptr comparand value = return $ AtomicCompareExchangeWeak ptr comparand value

atomicCompareExchangeWeakU :: Ptr space AtomicU32 -> Exp U32 -> Exp U32 -> ShaderM (Exp U32)
atomicCompareExchangeWeakU ptr comparand value = return $ AtomicCompareExchangeWeakU ptr comparand value

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

-- ============================================================================
-- Math Functions
-- ============================================================================

sqrt' :: Exp F32 -> Exp F32
sqrt' = Sqrt

abs' :: Exp a -> Exp a
abs' = Abs

min' :: Exp a -> Exp a -> Exp a
min' = Min

max' :: Exp a -> Exp a -> Exp a
max' = Max

exp' :: Exp F32 -> Exp F32
exp' = WGSL.AST.Exp

cos' :: Exp a -> Exp a
cos' = Cos

sin' :: Exp a -> Exp a
sin' = Sin

pow' :: Exp a -> Exp a -> Exp a
pow' = Pow

tanh' :: Exp a -> Exp a
tanh' = Tanh

clamp' :: Exp a -> Exp a -> Exp a -> Exp a
clamp' = Clamp
