{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}

module WGSL.AST where

import GHC.TypeLits (Nat)
import GHC.Generics
import Data.Proxy

-- | Phantom Types for WGSL types
data F32
data F16
data I32
data U32
data Bool_  -- Distinct from Haskell Bool

data Vec2 a
data Vec3 a
data Vec4 a

data Array (n :: Nat) a

-- | Struct type - phantom type representing a user-defined struct
-- The parameter 'a' is the Haskell type that corresponds to the WGSL struct
data Struct a

-- | Multi-dimensional views for safe array indexing
-- These are logical views over 1D storage buffers
data View1D (size :: Nat) a              -- [size]
data View2D (rows :: Nat) (cols :: Nat) a  -- [rows][cols]
data View3D (d1 :: Nat) (d2 :: Nat) (d3 :: Nat) a  -- [d1][d2][d3]

-- | Subgroup Matrix Types (for chromium_experimental_subgroup_matrix)
data SubgroupMatrixLeft (precision :: *) (m :: Nat) (n :: Nat)
data SubgroupMatrixRight (precision :: *) (m :: Nat) (n :: Nat)
data SubgroupMatrixResult (precision :: *) (m :: Nat) (n :: Nat)

-- | Memory Address Space Tags
data Storage    -- Global memory (Large, Slow)
data Workgroup  -- Shared memory (Fast, Inter-thread communication)
data Private    -- Local variables/Registers

-- | Typed Pointer ensuring space safety
newtype Ptr space a = Ptr String
  deriving (Show, Eq)

-- | Multi-dimensional view wrapper
-- Provides safe indexed access to buffers with automatic offset calculation
data View space a where
  -- 1D view: just wraps the pointer with size info
  View1D :: Ptr space (Array n elem) -> Int -> View space (View1D n elem)

  -- 2D view: tracks rows, cols, and row stride
  View2D :: Ptr space (Array n elem) -> Int -> Int -> Int -> View space (View2D rows cols elem)
  --        base pointer                rows   cols   rowStride

  -- 3D view: tracks three dimensions and strides
  View3D :: Ptr space (Array n elem) -> Int -> Int -> Int -> Int -> Int -> View space (View3D d1 d2 d3 elem)
  --        base pointer                d1     d2     d3     stride2  stride3

deriving instance Show (View space a)

-- | The AST Expression with Phantom Types
data Exp a where
  -- Literals
  LitF32 :: Float -> Exp F32
  LitF16 :: Float -> Exp F16
  LitI32 :: Int -> Exp I32
  LitU32 :: Int -> Exp U32
  LitBool :: Bool -> Exp Bool_

  -- Variables
  Var :: String -> Exp a

  -- Arithmetic
  Add :: Exp a -> Exp a -> Exp a
  Sub :: Exp a -> Exp a -> Exp a
  Mul :: Exp a -> Exp a -> Exp a
  Div :: Exp a -> Exp a -> Exp a
  Neg :: Exp a -> Exp a

  -- Comparison
  Eq :: Exp a -> Exp a -> Exp Bool_
  Ne :: Exp a -> Exp a -> Exp Bool_
  Lt :: Exp a -> Exp a -> Exp Bool_
  Le :: Exp a -> Exp a -> Exp Bool_
  Gt :: Exp a -> Exp a -> Exp Bool_
  Ge :: Exp a -> Exp a -> Exp Bool_

  -- Boolean Logic
  And :: Exp Bool_ -> Exp Bool_ -> Exp Bool_
  Or  :: Exp Bool_ -> Exp Bool_ -> Exp Bool_
  Not :: Exp Bool_ -> Exp Bool_

  -- Array/Pointer Accessors
  Index :: Exp (Array n a) -> Exp I32 -> Exp a
  PtrIndex :: Ptr s (Array n a) -> Exp I32 -> Exp a
  Deref :: Ptr s a -> Exp a

  -- Vector Accessors
  VecX :: Exp (Vec3 a) -> Exp a
  VecY :: Exp (Vec3 a) -> Exp a
  VecZ :: Exp (Vec3 a) -> Exp a

  -- Struct Field Access
  -- Represents: expr.fieldName
  FieldAccess :: Exp (Struct s) -> String -> Exp a

  -- Built-in Functions
  Sqrt :: Exp F32 -> Exp F32
  Abs  :: Exp a -> Exp a
  Min  :: Exp a -> Exp a -> Exp a
  Max  :: Exp a -> Exp a -> Exp a

  -- Type Conversion
  F32ToI32 :: Exp F32 -> Exp I32
  I32ToF32 :: Exp I32 -> Exp F32
  U32ToI32 :: Exp U32 -> Exp I32
  I32ToU32 :: Exp I32 -> Exp U32

  -- Subgroup Matrix Operations
  SubgroupMatrixLoad :: TypeRep -> Ptr s a -> Exp U32 -> Exp Bool_ -> Exp U32 -> Exp b
  SubgroupMatrixMultiplyAccumulate :: Exp a -> Exp b -> Exp c -> Exp c

deriving instance Show (Exp a)

-- | Statements (Imperative Actions)
data Stmt where
  -- Variable Declaration
  DeclVar :: String -> TypeRep -> Maybe ExpSome -> Stmt

  -- Assignment
  Assign :: String -> ExpSome -> Stmt
  PtrAssign :: Ptr s a -> Exp a -> Stmt

  -- Control Flow
  If :: Exp Bool_ -> [Stmt] -> [Stmt] -> Stmt  -- Condition, Then, Else
  While :: Exp Bool_ -> [Stmt] -> Stmt
  For :: String -> Exp I32 -> Exp I32 -> Maybe (Exp I32) -> [Stmt] -> Stmt  -- var, start, end, optional step, body

  -- Synchronization
  Barrier :: Stmt  -- workgroupBarrier()

  -- Subgroup Matrix Store
  SubgroupMatrixStore :: Ptr s a -> Exp U32 -> Exp b -> Exp Bool_ -> Exp U32 -> Stmt

  -- Return
  Return :: ExpSome -> Stmt

  -- Comment (for debugging/documentation)
  Comment :: String -> Stmt

  -- Raw WGSL statement (for special cases not covered by DSL)
  RawStmt :: String -> Stmt

deriving instance Show Stmt

-- | Type-erased expression for heterogeneous lists
data ExpSome where
  SomeExp :: Exp a -> ExpSome

deriving instance Show ExpSome

-- | WGSL Type Representation (for code generation)
data TypeRep
  = TF32
  | TF16
  | TI32
  | TU32
  | TBool
  | TVec2 TypeRep
  | TVec3 TypeRep
  | TVec4 TypeRep
  | TArray Int TypeRep  -- size, element type
  | TPtr MemorySpace TypeRep
  | TSubgroupMatrixLeft TypeRep Int Int    -- precision, m, n
  | TSubgroupMatrixRight TypeRep Int Int   -- precision, m, n
  | TSubgroupMatrixResult TypeRep Int Int  -- precision, m, n
  | TStruct String  -- Struct name (references a struct definition)
  deriving (Show, Eq)

data MemorySpace
  = MStorage
  | MWorkgroup
  | MPrivate
  deriving (Show, Eq)

-- | Struct Definition
data StructDef = StructDef
  { structName :: String
  , structFields :: [(String, TypeRep)]  -- field name, field type
  }
  deriving (Show, Eq)

-- | Function Declaration
data FunctionDecl = FunctionDecl
  { funcName :: String
  , funcParams :: [(Maybe String, String, TypeRep)]  -- (optional @builtin(xxx), name, type)
  , funcReturnType :: Maybe TypeRep
  , funcBody :: [Stmt]
  , funcAttributes :: [String]  -- e.g., "@compute", "@workgroup_size(256)"
  }
  deriving (Show)

-- | Shader Module (Top-level)
data ShaderModule = ShaderModule
  { moduleFunctions :: [FunctionDecl]
  , moduleGlobals :: [(String, TypeRep, MemorySpace)]  -- var<storage>, var<workgroup>
  , moduleStructs :: [StructDef]  -- struct definitions
  , moduleExtensions :: [String]  -- enable directives (e.g., "f16", "chromium_experimental_subgroup_matrix")
  , moduleDiagnostics :: [String]  -- diagnostic directives (e.g., "off, chromium.subgroup_matrix_uniformity")
  , moduleBindings :: [(String, Int)]  -- buffer name -> binding index mapping (for safe named execution)
  }
  deriving (Show)

-- | WGSLType typeclass for automatic struct generation from Haskell types
-- Use GHC.Generics to derive WGSL struct definitions automatically
class WGSLType a where
  -- | Get the WGSL type name for this type
  wgslTypeName :: Proxy a -> String

  -- | Get the TypeRep for this type
  wgslTypeRep :: Proxy a -> TypeRep

  -- | Generate a struct definition for this type (if it's a struct)
  -- Returns Nothing for primitive types, Just StructDef for user-defined structs
  wgslStructDef :: Proxy a -> Maybe StructDef

  default wgslTypeName :: (Generic a, GWGSLType (Rep a)) => Proxy a -> String
  wgslTypeName _ = gwgslTypeName (Proxy :: Proxy (Rep a ()))

  default wgslStructDef :: (Generic a, GWGSLType (Rep a)) => Proxy a -> Maybe StructDef
  wgslStructDef proxy =
    let name = wgslTypeName proxy
        fields = gwgslStructFields (Proxy :: Proxy (Rep a ()))
    in Just (StructDef name fields)

-- | Generic typeclass for deriving WGSLType instances
class GWGSLType f where
  gwgslTypeName :: Proxy (f a) -> String
  gwgslStructFields :: Proxy (f a) -> [(String, TypeRep)]

-- | Instance for datatype metadata (constructor name)
instance (Datatype d, GWGSLType f) => GWGSLType (D1 d f) where
  gwgslTypeName _ = datatypeName (undefined :: D1 d f a)
  gwgslStructFields _ = gwgslStructFields (Proxy :: Proxy (f a))

-- | Instance for constructor metadata
instance (GWGSLType f) => GWGSLType (C1 c f) where
  gwgslTypeName _ = gwgslTypeName (Proxy :: Proxy (f a))
  gwgslStructFields _ = gwgslStructFields (Proxy :: Proxy (f a))

-- | Instance for selector (field)
instance (Selector s, WGSLType t) => GWGSLType (S1 s (K1 i t)) where
  gwgslTypeName _ = wgslTypeName (Proxy :: Proxy t)
  gwgslStructFields _ = [(selName (undefined :: S1 s (K1 i t) a), wgslTypeRep (Proxy :: Proxy t))]

-- | Instance for product (multiple fields)
instance (GWGSLType f, GWGSLType g) => GWGSLType (f :*: g) where
  gwgslTypeName _ = gwgslTypeName (Proxy :: Proxy (f a))
  gwgslStructFields _ =
    gwgslStructFields (Proxy :: Proxy (f a)) ++ gwgslStructFields (Proxy :: Proxy (g a))

-- | Primitive type instances
instance WGSLType F32 where
  wgslTypeName _ = "f32"
  wgslTypeRep _ = TF32
  wgslStructDef _ = Nothing

instance WGSLType F16 where
  wgslTypeName _ = "f16"
  wgslTypeRep _ = TF16
  wgslStructDef _ = Nothing

instance WGSLType I32 where
  wgslTypeName _ = "i32"
  wgslTypeRep _ = TI32
  wgslStructDef _ = Nothing

instance WGSLType U32 where
  wgslTypeName _ = "u32"
  wgslTypeRep _ = TU32
  wgslStructDef _ = Nothing

instance WGSLType Bool_ where
  wgslTypeName _ = "bool"
  wgslTypeRep _ = TBool
  wgslStructDef _ = Nothing

-- | Vec instances
instance WGSLType a => WGSLType (Vec2 a) where
  wgslTypeName _ = "vec2<" ++ wgslTypeName (Proxy :: Proxy a) ++ ">"
  wgslTypeRep _ = TVec2 (wgslTypeRep (Proxy :: Proxy a))
  wgslStructDef _ = Nothing

instance WGSLType a => WGSLType (Vec3 a) where
  wgslTypeName _ = "vec3<" ++ wgslTypeName (Proxy :: Proxy a) ++ ">"
  wgslTypeRep _ = TVec3 (wgslTypeRep (Proxy :: Proxy a))
  wgslStructDef _ = Nothing
