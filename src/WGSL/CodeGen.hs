{-# LANGUAGE GADTs #-}

module WGSL.CodeGen
  ( prettyExp
  , prettyStmt
  , prettyTypeRep
  , prettyFunction
  , prettyModule
  , generateWGSL
  , generateWGSLOptimized
  ) where

import WGSL.AST
import WGSL.Optimize
import Data.List (intercalate)

-- | Pretty-print an expression
prettyExp :: Exp a -> String
prettyExp expr = case expr of
  LitF32 f -> show f ++ "f"
  LitF16 f -> show f ++ "h"
  LitI32 i -> show i
  LitU32 i -> show i ++ "u"
  LitBool True -> "true"
  LitBool False -> "false"

  Var name -> name

  Add a b -> "(" ++ prettyExp a ++ " + " ++ prettyExp b ++ ")"
  Sub a b -> "(" ++ prettyExp a ++ " - " ++ prettyExp b ++ ")"
  Mul a b -> "(" ++ prettyExp a ++ " * " ++ prettyExp b ++ ")"
  Div a b -> "(" ++ prettyExp a ++ " / " ++ prettyExp b ++ ")"
  Mod a b -> "(" ++ prettyExp a ++ " % " ++ prettyExp b ++ ")"
  Neg a -> "(-" ++ prettyExp a ++ ")"

  Eq a b -> "(" ++ prettyExp a ++ " == " ++ prettyExp b ++ ")"
  Ne a b -> "(" ++ prettyExp a ++ " != " ++ prettyExp b ++ ")"
  Lt a b -> "(" ++ prettyExp a ++ " < " ++ prettyExp b ++ ")"
  Le a b -> "(" ++ prettyExp a ++ " <= " ++ prettyExp b ++ ")"
  Gt a b -> "(" ++ prettyExp a ++ " > " ++ prettyExp b ++ ")"
  Ge a b -> "(" ++ prettyExp a ++ " >= " ++ prettyExp b ++ ")"

  And a b -> "(" ++ prettyExp a ++ " && " ++ prettyExp b ++ ")"
  Or a b -> "(" ++ prettyExp a ++ " || " ++ prettyExp b ++ ")"
  Not a -> "(!" ++ prettyExp a ++ ")"

  Index arr idx -> prettyExp arr ++ "[" ++ prettyExp idx ++ "]"
  PtrIndex (Ptr name) idx -> name ++ "[" ++ prettyExp idx ++ "]"
  Deref (Ptr name) -> name

  VecX v -> prettyExp v ++ ".x"
  VecY v -> prettyExp v ++ ".y"
  VecZ v -> prettyExp v ++ ".z"

  FieldAccess structExpr fieldName -> prettyExp structExpr ++ "." ++ fieldName

  Sqrt a -> "sqrt(" ++ prettyExp a ++ ")"
  Abs a -> "abs(" ++ prettyExp a ++ ")"
  Min a b -> "min(" ++ prettyExp a ++ ", " ++ prettyExp b ++ ")"
  Max a b -> "max(" ++ prettyExp a ++ ", " ++ prettyExp b ++ ")"
  WGSL.AST.Exp a -> "exp(" ++ prettyExp a ++ ")"
  Cos a -> "cos(" ++ prettyExp a ++ ")"
  Sin a -> "sin(" ++ prettyExp a ++ ")"
  Pow a b -> "pow(" ++ prettyExp a ++ ", " ++ prettyExp b ++ ")"
  Tanh a -> "tanh(" ++ prettyExp a ++ ")"
  Clamp x minVal maxVal -> "clamp(" ++ prettyExp x ++ ", " ++ prettyExp minVal ++ ", " ++ prettyExp maxVal ++ ")"

  F32ToI32 a -> "i32(" ++ prettyExp a ++ ")"
  I32ToF32 a -> "f32(" ++ prettyExp a ++ ")"
  U32ToI32 a -> "i32(" ++ prettyExp a ++ ")"
  I32ToU32 a -> "u32(" ++ prettyExp a ++ ")"
  F16ToF32 a -> "f32(" ++ prettyExp a ++ ")"
  F32ToF16 a -> "f16(" ++ prettyExp a ++ ")"
  I32ToF16 a -> "f16(" ++ prettyExp a ++ ")"

  ShiftLeft a b -> "(" ++ prettyExp a ++ " << " ++ prettyExp b ++ ")"
  ShiftRight a b -> "(" ++ prettyExp a ++ " >> " ++ prettyExp b ++ ")"
  BitAnd a b -> "(" ++ prettyExp a ++ " & " ++ prettyExp b ++ ")"
  BitOr a b -> "(" ++ prettyExp a ++ " | " ++ prettyExp b ++ ")"
  BitXor a b -> "(" ++ prettyExp a ++ " ^ " ++ prettyExp b ++ ")"

  SubgroupMatrixLoad ty (Ptr name) offset transpose stride ->
    "subgroupMatrixLoad<" ++ prettyTypeRep ty ++ ">(&" ++ name ++ ", " ++
    prettyExp offset ++ ", " ++ prettyExp transpose ++ ", " ++ prettyExp stride ++ ")"

  SubgroupMatrixMultiplyAccumulate left right acc ->
    "subgroupMatrixMultiplyAccumulate(" ++ prettyExp left ++ ", " ++
    prettyExp right ++ ", " ++ prettyExp acc ++ ")"

  -- Texture Operations
  TextureSample texture sampler uv ->
    "textureSample(" ++ prettyExp texture ++ ", " ++ prettyExp sampler ++ ", " ++
    prettyExp uv ++ ")"

  TextureLoad texture coords mipLevel ->
    "textureLoad(" ++ prettyExp texture ++ ", " ++ prettyExp coords ++ ", " ++
    prettyExp mipLevel ++ ")"

  -- Atomic Operations
  AtomicAdd (Ptr name) value ->
    "atomicAdd(&" ++ name ++ ", " ++ prettyExp value ++ ")"
  AtomicAddU (Ptr name) value ->
    "atomicAdd(&" ++ name ++ ", " ++ prettyExp value ++ ")"

  AtomicSub (Ptr name) value ->
    "atomicSub(&" ++ name ++ ", " ++ prettyExp value ++ ")"
  AtomicSubU (Ptr name) value ->
    "atomicSub(&" ++ name ++ ", " ++ prettyExp value ++ ")"

  AtomicMin (Ptr name) value ->
    "atomicMin(&" ++ name ++ ", " ++ prettyExp value ++ ")"
  AtomicMinU (Ptr name) value ->
    "atomicMin(&" ++ name ++ ", " ++ prettyExp value ++ ")"

  AtomicMax (Ptr name) value ->
    "atomicMax(&" ++ name ++ ", " ++ prettyExp value ++ ")"
  AtomicMaxU (Ptr name) value ->
    "atomicMax(&" ++ name ++ ", " ++ prettyExp value ++ ")"

  AtomicExchange (Ptr name) value ->
    "atomicExchange(&" ++ name ++ ", " ++ prettyExp value ++ ")"
  AtomicExchangeU (Ptr name) value ->
    "atomicExchange(&" ++ name ++ ", " ++ prettyExp value ++ ")"

  AtomicCompareExchangeWeak (Ptr name) comparand value ->
    "atomicCompareExchangeWeak(&" ++ name ++ ", " ++ prettyExp comparand ++ ", " ++
    prettyExp value ++ ").old_value"
  AtomicCompareExchangeWeakU (Ptr name) comparand value ->
    "atomicCompareExchangeWeak(&" ++ name ++ ", " ++ prettyExp comparand ++ ", " ++
    prettyExp value ++ ").old_value"

prettyExpSome :: ExpSome -> String
prettyExpSome (SomeExp e) = prettyExp e

-- | Pretty-print a type
prettyTypeRep :: TypeRep -> String
prettyTypeRep ty = case ty of
  TF32 -> "f32"
  TF16 -> "f16"
  TI32 -> "i32"
  TU32 -> "u32"
  TBool -> "bool"
  TVec2 t -> "vec2<" ++ prettyTypeRep t ++ ">"
  TVec3 t -> "vec3<" ++ prettyTypeRep t ++ ">"
  TVec4 t -> "vec4<" ++ prettyTypeRep t ++ ">"
  TArray n t -> "array<" ++ prettyTypeRep t ++ ", " ++ show n ++ ">"
  TPtr space t -> "ptr<" ++ prettyMemorySpace space ++ ", " ++ prettyTypeRep t ++ ">"
  TSubgroupMatrixLeft precision m n ->
    "subgroup_matrix_left<" ++ prettyTypeRep precision ++ ", " ++ show m ++ ", " ++ show n ++ ">"
  TSubgroupMatrixRight precision m n ->
    "subgroup_matrix_right<" ++ prettyTypeRep precision ++ ", " ++ show m ++ ", " ++ show n ++ ">"
  TSubgroupMatrixResult precision m n ->
    "subgroup_matrix_result<" ++ prettyTypeRep precision ++ ", " ++ show m ++ ", " ++ show n ++ ">"
  TStruct name -> name  -- Just the struct name
  TTexture2D format -> "texture_2d<" ++ format ++ ">"
  TSampler -> "sampler"
  TAtomicI32 -> "atomic<i32>"
  TAtomicU32 -> "atomic<u32>"

prettyMemorySpace :: MemorySpace -> String
prettyMemorySpace MStorage = "storage"
prettyMemorySpace MWorkgroup = "workgroup"
prettyMemorySpace MPrivate = "private"

-- | Pretty-print a statement with indentation
prettyStmt :: Int -> Stmt -> String
prettyStmt indent stmt =
  let ind = replicate (indent * 2) ' '
  in case stmt of
    DeclVar name ty Nothing ->
      ind ++ "var " ++ name ++ ": " ++ prettyTypeRep ty ++ ";"

    DeclVar name ty (Just expr) ->
      ind ++ "var " ++ name ++ ": " ++ prettyTypeRep ty ++ " = " ++ prettyExpSome expr ++ ";"

    Assign name expr ->
      ind ++ name ++ " = " ++ prettyExpSome expr ++ ";"

    PtrAssign (Ptr name) expr ->
      ind ++ name ++ " = " ++ prettyExp expr ++ ";"

    If cond thenStmts [] ->
      ind ++ "if (" ++ prettyExp cond ++ ") {\n" ++
      concatMap (prettyStmt (indent + 1)) thenStmts ++
      ind ++ "}\n"

    If cond thenStmts elseStmts ->
      ind ++ "if (" ++ prettyExp cond ++ ") {\n" ++
      concatMap (prettyStmt (indent + 1)) thenStmts ++
      ind ++ "} else {\n" ++
      concatMap (prettyStmt (indent + 1)) elseStmts ++
      ind ++ "}\n"

    While cond bodyStmts ->
      ind ++ "while (" ++ prettyExp cond ++ ") {\n" ++
      concatMap (prettyStmt (indent + 1)) bodyStmts ++
      ind ++ "}\n"

    For varName start end mbStep bodyStmts ->
      let stepExpr = case mbStep of
            Nothing -> varName ++ "++"
            Just step -> varName ++ " = " ++ varName ++ " + " ++ prettyExp step
      in ind ++ "for (var " ++ varName ++ " = " ++ prettyExp start ++
         "; " ++ varName ++ " < " ++ prettyExp end ++
         "; " ++ stepExpr ++ ") {\n" ++
         concatMap (prettyStmt (indent + 1)) bodyStmts ++
         ind ++ "}\n"

    Barrier ->
      ind ++ "workgroupBarrier();\n"

    SubgroupMatrixStore (Ptr name) offset matrix transpose stride ->
      ind ++ "subgroupMatrixStore(&" ++ name ++ ", " ++
      prettyExp offset ++ ", " ++ prettyExp matrix ++ ", " ++
      prettyExp transpose ++ ", " ++ prettyExp stride ++ ");\n"

    TextureStore texture coords value ->
      ind ++ "textureStore(" ++ prettyExp texture ++ ", " ++
      prettyExp coords ++ ", " ++ prettyExp value ++ ");\n"

    Return expr ->
      ind ++ "return " ++ prettyExpSome expr ++ ";\n"

    Comment text ->
      ind ++ "// " ++ text ++ "\n"

    RawStmt code ->
      ind ++ code ++ "\n"

-- | Pretty-print a function declaration
prettyFunction :: FunctionDecl -> String
prettyFunction func =
  let attrs = concatMap (\a -> a ++ "\n") (funcAttributes func)
      params = intercalate ", " $
        map (\(mbBuiltin, name, ty) ->
          case mbBuiltin of
            Just builtin -> "@builtin(" ++ builtin ++ ") " ++ name ++ ": " ++ prettyTypeRep ty
            Nothing -> name ++ ": " ++ prettyTypeRep ty
        ) (funcParams func)
      retType = case funcReturnType func of
        Nothing -> ""
        Just ty -> " -> " ++ prettyTypeRep ty
  in attrs ++
     "fn " ++ funcName func ++ "(" ++ params ++ ")" ++ retType ++ " {\n" ++
     concatMap (prettyStmt 1) (funcBody func) ++
     "}\n"

-- | Pretty-print a shader module
prettyModule :: ShaderModule -> String
prettyModule mod =
  let extensions = concatMap (\ext -> "enable " ++ ext ++ ";\n") (moduleExtensions mod)
      diagnostics = concatMap (\diag -> "diagnostic (" ++ diag ++ ");\n") (moduleDiagnostics mod)
      extensionSection = if null extensions && null diagnostics then "" else extensions ++ diagnostics ++ "\n"
      structs = concatMap prettyStruct (moduleStructs mod)
      structSection = if null structs then "" else structs ++ "\n"
      (storageVars, workgroupVars) = partitionGlobals (moduleGlobals mod)
      storageDecls = zipWith prettyStorageGlobal [0..] storageVars
      workgroupDecls = map prettyWorkgroupGlobal workgroupVars
      functions = concatMap prettyFunction (moduleFunctions mod)
  in extensionSection ++ structSection ++ concat storageDecls ++ concat workgroupDecls ++ "\n" ++ functions
  where
    partitionGlobals = foldr partition ([], [])
    partition g@(_, _, MStorage) (s, w) = (g:s, w)
    partition g@(_, _, MWorkgroup) (s, w) = (s, g:w)
    partition g@(_, _, MPrivate) (s, w) = (g:s, w)  -- Private vars go with storage

prettyStorageGlobal :: Int -> (String, TypeRep, MemorySpace) -> String
prettyStorageGlobal bindingIndex (name, ty, space) =
  "@group(0) @binding(" ++ show bindingIndex ++ ")\n" ++
  "var<" ++ prettyMemorySpace space ++ ", read_write> " ++ name ++ ": " ++ prettyTypeRep ty ++ ";\n"

prettyWorkgroupGlobal :: (String, TypeRep, MemorySpace) -> String
prettyWorkgroupGlobal (name, ty, _) =
  "var<workgroup> " ++ name ++ ": " ++ prettyTypeRep ty ++ ";\n"

-- | Pretty-print a struct definition
prettyStruct :: StructDef -> String
prettyStruct structDef =
  "struct " ++ structName structDef ++ " {\n" ++
  concatMap prettyField (structFields structDef) ++
  "}\n"
  where
    prettyField (fieldName, fieldType) =
      "  " ++ fieldName ++ ": " ++ prettyTypeRep fieldType ++ ",\n"

-- | Generate WGSL code from a shader module
generateWGSL :: ShaderModule -> String
generateWGSL = prettyModule

-- | Generate WGSL code with optimization (constant folding, etc.)
generateWGSLOptimized :: ShaderModule -> String
generateWGSLOptimized = prettyModule . optimizeModule
