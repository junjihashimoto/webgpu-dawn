{-# LANGUAGE GADTs #-}

{-|
WGSL AST Optimization

Implements simple optimizations like constant folding and algebraic simplifications
to generate cleaner WGSL code.

Optimizations performed:
- Constant folding: Add (Lit 3) (Lit 5) -> Lit 8
- Identity elimination: Mul x (Lit 1) -> x, Add x (Lit 0) -> x
- Zero elimination: Mul x (Lit 0) -> Lit 0
-}

module WGSL.Optimize
  ( optimizeExp
  , optimizeStmt
  , optimizeModule
  ) where

import WGSL.AST
import Prelude (Bool(..), Maybe(..), not, div, (==), (/=), (+), (-), (*), (/), ($), (.), map, fmap, (&&))
import qualified Prelude as P

-- | Optimize an expression (constant folding + algebraic simplifications)
optimizeExp :: Exp a -> Exp a
optimizeExp expr = case expr of
  -- Constant folding for arithmetic
  Add (LitI32 a) (LitI32 b) -> LitI32 (a P.+ b)
  Add (LitU32 a) (LitU32 b) -> LitU32 (a P.+ b)
  Add (LitF32 a) (LitF32 b) -> LitF32 (a P.+ b)
  Add (LitF16 a) (LitF16 b) -> LitF16 (a P.+ b)

  Sub (LitI32 a) (LitI32 b) -> LitI32 (a P.- b)
  Sub (LitU32 a) (LitU32 b) -> LitU32 (a P.- b)
  Sub (LitF32 a) (LitF32 b) -> LitF32 (a P.- b)
  Sub (LitF16 a) (LitF16 b) -> LitF16 (a P.- b)

  Mul (LitI32 a) (LitI32 b) -> LitI32 (a P.* b)
  Mul (LitU32 a) (LitU32 b) -> LitU32 (a P.* b)
  Mul (LitF32 a) (LitF32 b) -> LitF32 (a P.* b)
  Mul (LitF16 a) (LitF16 b) -> LitF16 (a P.* b)

  Div (LitI32 a) (LitI32 b) | b P./= 0 -> LitI32 (a `P.div` b)
  Div (LitU32 a) (LitU32 b) | b P./= 0 -> LitU32 (a `P.div` b)
  Div (LitF32 a) (LitF32 b) | b P./= 0 -> LitF32 (a P./ b)
  Div (LitF16 a) (LitF16 b) | b P./= 0 -> LitF16 (a P./ b)

  -- Identity: x + 0 = x
  Add x (LitI32 0) -> optimizeExp x
  Add x (LitU32 0) -> optimizeExp x
  Add x (LitF32 0) -> optimizeExp x
  Add x (LitF16 0) -> optimizeExp x
  Add (LitI32 0) x -> optimizeExp x
  Add (LitU32 0) x -> optimizeExp x
  Add (LitF32 0) x -> optimizeExp x
  Add (LitF16 0) x -> optimizeExp x

  -- Identity: x * 1 = x
  Mul x (LitI32 1) -> optimizeExp x
  Mul x (LitU32 1) -> optimizeExp x
  Mul x (LitF32 1) -> optimizeExp x
  Mul x (LitF16 1) -> optimizeExp x
  Mul (LitI32 1) x -> optimizeExp x
  Mul (LitU32 1) x -> optimizeExp x
  Mul (LitF32 1) x -> optimizeExp x
  Mul (LitF16 1) x -> optimizeExp x

  -- Zero: x * 0 = 0
  Mul _ (LitI32 0) -> LitI32 0
  Mul _ (LitU32 0) -> LitU32 0
  Mul _ (LitF32 0) -> LitF32 0
  Mul _ (LitF16 0) -> LitF16 0
  Mul (LitI32 0) _ -> LitI32 0
  Mul (LitU32 0) _ -> LitU32 0
  Mul (LitF32 0) _ -> LitF32 0
  Mul (LitF16 0) _ -> LitF16 0

  -- Identity: x - 0 = x
  Sub x (LitI32 0) -> optimizeExp x
  Sub x (LitU32 0) -> optimizeExp x
  Sub x (LitF32 0) -> optimizeExp x
  Sub x (LitF16 0) -> optimizeExp x

  -- Recursive optimization for compound expressions
  Add a b ->
    let a' = optimizeExp a
        b' = optimizeExp b
    in if sameExp a a' && sameExp b b'
       then Add a b
       else optimizeExp (Add a' b')

  Sub a b ->
    let a' = optimizeExp a
        b' = optimizeExp b
    in if sameExp a a' && sameExp b b'
       then Sub a b
       else optimizeExp (Sub a' b')

  Mul a b ->
    let a' = optimizeExp a
        b' = optimizeExp b
    in if sameExp a a' && sameExp b b'
       then Mul a b
       else optimizeExp (Mul a' b')

  Div a b ->
    let a' = optimizeExp a
        b' = optimizeExp b
    in if sameExp a a' && sameExp b b'
       then Div a b
       else optimizeExp (Div a' b')

  Neg a -> Neg (optimizeExp a)
  Abs a -> Abs (optimizeExp a)

  -- Boolean operations
  And (LitBool True) x -> optimizeExp x
  And x (LitBool True) -> optimizeExp x
  And (LitBool False) _ -> LitBool False
  And _ (LitBool False) -> LitBool False

  Or (LitBool True) _ -> LitBool True
  Or _ (LitBool True) -> LitBool True
  Or (LitBool False) x -> optimizeExp x
  Or x (LitBool False) -> optimizeExp x

  Not (LitBool b) -> LitBool (P.not b)
  Not (Not x) -> optimizeExp x  -- Double negation

  -- Comparison with same operands
  Eq a b | sameExp a b -> LitBool True
  Lt a b | sameExp a b -> LitBool False
  Le a b | sameExp a b -> LitBool True
  Gt a b | sameExp a b -> LitBool False
  Ge a b | sameExp a b -> LitBool True

  -- Default: no optimization
  _ -> expr

-- | Check if two expressions are syntactically the same
sameExp :: Exp a -> Exp b -> Bool
sameExp (LitI32 a) (LitI32 b) = a P.== b
sameExp (LitU32 a) (LitU32 b) = a P.== b
sameExp (LitF32 a) (LitF32 b) = a P.== b
sameExp (LitF16 a) (LitF16 b) = a P.== b
sameExp (LitBool a) (LitBool b) = a P.== b
sameExp (Var a) (Var b) = a P.== b
sameExp _ _ = False

-- | Optimize a statement
optimizeStmt :: Stmt -> Stmt
optimizeStmt stmt = case stmt of
  Assign name (SomeExp expr) -> Assign name (SomeExp (optimizeExp expr))

  DeclVar name ty (Just (SomeExp expr)) ->
    DeclVar name ty (Just (SomeExp (optimizeExp expr)))

  If cond thenStmts elseStmts ->
    let cond' = optimizeExp cond
        thenStmts' = map optimizeStmt thenStmts
        elseStmts' = map optimizeStmt elseStmts
    in case cond' of
         LitBool True -> If cond' thenStmts' []  -- else branch unreachable
         LitBool False -> If cond' [] elseStmts' -- then branch unreachable
         _ -> If cond' thenStmts' elseStmts'

  While cond body ->
    While (optimizeExp cond) (map optimizeStmt body)

  For var start end mbStep body ->
    For var (optimizeExp start) (optimizeExp end) (fmap optimizeExp mbStep) (map optimizeStmt body)

  SubgroupMatrixStore ptr offset mat transpose stride ->
    SubgroupMatrixStore ptr (optimizeExp offset) (optimizeExp mat)
                        (optimizeExp transpose) (optimizeExp stride)

  Comment _ -> stmt  -- Comments pass through unchanged

  RawStmt _ -> stmt  -- Raw statements pass through unchanged

  _ -> stmt

-- | Optimize a complete shader module
optimizeModule :: ShaderModule -> ShaderModule
optimizeModule mod = mod
  { moduleFunctions = map optimizeFunction (moduleFunctions mod)
  }

-- | Optimize a function
optimizeFunction :: FunctionDecl -> FunctionDecl
optimizeFunction func = func
  { funcBody = map optimizeStmt (funcBody func)
  }
