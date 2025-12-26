{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module WGSL.Monad
  ( ShaderM
  , ShaderState(..)
  , runShader
  , emitStmt
  , freshVar
  , captureStatements

  -- Variable operations
  , var
  , varNamed
  , shared
  , sharedNamed

  -- Assignment
  , assign
  , (<==)

  -- Control Flow
  , if_
  , while_
  , for_
  , forStep_
  , loop  -- HOAS-style loop

  -- Synchronization
  , barrier

  -- Built-ins
  , globalId
  , localId
  , workgroupId
  , numWorkgroups

  -- Reading/Writing buffers
  , readPtr
  , globalBuffer
  , readBuffer
  , writeBuffer

  -- Automatic binding management
  , declareInputBuffer
  , declareOutputBuffer
  , declareStorageBuffer

  -- Multi-dimensional views (safe array indexing)
  , makeView1D
  , makeView2D
  , makeView3D
  , viewIndex1D
  , viewIndex2D
  , viewIndex3D
  , readView1D
  , readView2D
  , readView3D
  , writeView1D
  , writeView2D
  , writeView3D

  -- Typed matrix views (HOAS-style, prevents index swapping)
  , RowIndex(..)
  , ColIndex(..)
  , MatrixView
  , makeMatrixView
  , at
  , writeAt

  -- Subgroup Matrix Operations
  , subgroupMatrixLoad
  , subgroupMatrixMultiplyAccumulate
  , subgroupMatrixStore

  -- High-level subgroup helpers
  , MatrixRole(..)
  , newSubgroupMatrix
  , newSubgroupMatrixZero
  , initializeSubgroupMatrix
  , loadMatrix
  , storeMatrix
  , mma

  -- High-level helpers
  , staticFor
  , staticFor_
  , litF
  , litI
  , litU

  -- Debug support
  , debugPrint
  , debugPrintF
  , debugPrintI
  , debugPrintU
  ) where

import WGSL.AST
import WGSL.CodeGen (prettyTypeRep)
import Control.Monad.State
import Prelude
import qualified Prelude as P

-- | Shader Construction State
data ShaderState = ShaderState
  { stmts :: [Stmt]           -- Accumulated statements
  , varCounter :: Int          -- For generating unique variable names
  , sharedVars :: [(String, TypeRep)]  -- Shared memory declarations
  , declaredBuffers :: [(String, TypeRep, MemorySpace)]  -- Auto-tracked buffer bindings
  }
  deriving (Show)

-- | The Shader Monad
newtype ShaderM a = ShaderM { unShaderM :: State ShaderState a }
  deriving (Functor, Applicative, Monad, MonadState ShaderState)

-- | Run a shader computation and extract the statements
runShader :: ShaderM a -> (a, ShaderState)
runShader m = runState (unShaderM m) initialState
  where
    initialState = ShaderState [] 0 [] []

-- | Emit a statement
emitStmt :: Stmt -> ShaderM ()
emitStmt stmt = modify $ \s -> s { stmts = stmts s ++ [stmt] }

-- | Generate a fresh variable name
freshVar :: String -> ShaderM String
freshVar prefix = do
  s <- get
  let n = varCounter s
  put s { varCounter = n + 1 }
  return $ prefix ++ show n

-- | Capture statements from a monadic action (for control flow)
captureStatements :: ShaderM a -> ShaderM (a, [Stmt])
captureStatements action = do
  oldState <- get
  put oldState { stmts = [] }  -- Clear statements
  result <- action
  newState <- get
  let capturedStmts = stmts newState
  -- Restore old state but keep var counter
  put oldState { varCounter = varCounter newState }
  return (result, capturedStmts)

-- | Declare a private variable
var :: TypeRep -> Exp a -> ShaderM (Ptr Private a)
var ty expr = do
  name <- freshVar "v"
  emitStmt $ DeclVar name ty (Just $ SomeExp expr)
  return $ Ptr name

-- | Declare a named private variable
varNamed :: String -> TypeRep -> Exp a -> ShaderM (Ptr Private a)
varNamed name ty expr = do
  emitStmt $ DeclVar name ty (Just $ SomeExp expr)
  return $ Ptr name

-- | Declare shared memory (workgroup-scoped)
shared :: TypeRep -> ShaderM (Ptr Workgroup a)
shared ty = do
  name <- freshVar "shared"
  modify $ \s -> s { sharedVars = sharedVars s ++ [(name, ty)] }
  return $ Ptr name

-- | Declare named shared memory
sharedNamed :: String -> TypeRep -> ShaderM (Ptr Workgroup a)
sharedNamed name ty = do
  modify $ \s -> s { sharedVars = sharedVars s ++ [(name, ty)] }
  return $ Ptr name

-- | Assignment operator
assign :: Ptr s a -> Exp a -> ShaderM ()
assign (Ptr name) expr = emitStmt $ Assign name (SomeExp expr)

-- | Infix assignment operator
(<==) :: Ptr s a -> Exp a -> ShaderM ()
(<==) = assign
infixr 0 <==

-- | If-then-else
if_ :: Exp Bool_ -> ShaderM () -> ShaderM () -> ShaderM ()
if_ cond thenBranch elseBranch = do
  (_, thenStmts) <- captureStatements thenBranch
  (_, elseStmts) <- captureStatements elseBranch
  emitStmt $ If cond thenStmts elseStmts

-- | While loop
while_ :: Exp Bool_ -> ShaderM () -> ShaderM ()
while_ cond body = do
  (_, bodyStmts) <- captureStatements body
  emitStmt $ While cond bodyStmts

-- | For loop (increments by 1)
for_ :: String -> Exp I32 -> Exp I32 -> ShaderM () -> ShaderM ()
for_ varName start end body = do
  (_, bodyStmts) <- captureStatements body
  emitStmt $ For varName start end Nothing bodyStmts

-- | For loop with custom step
forStep_ :: String -> Exp I32 -> Exp I32 -> Exp I32 -> ShaderM () -> ShaderM ()
forStep_ varName start end step body = do
  (_, bodyStmts) <- captureStatements body
  emitStmt $ For varName start end (Just step) bodyStmts

-- | HOAS-style loop: no string variable names, pass loop variable as Exp I32
-- Usage: loop start end step $ \i -> do { ... use i ... }
loop :: Exp I32 -> Exp I32 -> Exp I32 -> (Exp I32 -> ShaderM ()) -> ShaderM ()
loop start end step bodyFn = do
  varName <- freshVar "i"
  (_, bodyStmts) <- captureStatements (bodyFn (Var varName))
  emitStmt $ For varName start end (Just step) bodyStmts

-- | Workgroup barrier (synchronization)
barrier :: ShaderM ()
barrier = emitStmt Barrier

-- | Built-in: global invocation ID
globalId :: ShaderM (Exp (Vec3 U32))
globalId = return $ Var "global_invocation_id"

-- | Built-in: local invocation ID
localId :: ShaderM (Exp (Vec3 U32))
localId = return $ Var "local_invocation_id"

-- | Built-in: workgroup ID
workgroupId :: ShaderM (Exp (Vec3 U32))
workgroupId = return $ Var "workgroup_id"

-- | Built-in: number of workgroups
numWorkgroups :: ShaderM (Exp (Vec3 U32))
numWorkgroups = return $ Var "num_workgroups"

-- | Read from a pointer
readPtr :: Ptr s a -> ShaderM (Exp a)
readPtr ptr = return $ Deref ptr

-- | Create a reference to a global storage buffer by name
-- This allows accessing buffers declared in moduleGlobals
globalBuffer :: String -> Ptr Storage a
globalBuffer name = Ptr name

-- | Index into a global array buffer
readBuffer :: Ptr Storage (Array n a) -> Exp I32 -> ShaderM (Exp a)
readBuffer ptr idx = return $ PtrIndex ptr idx

-- | Write to a buffer index
writeBuffer :: Ptr Storage (Array n a) -> Exp I32 -> Exp a -> ShaderM ()
writeBuffer (Ptr name) idx value =
  emitStmt $ Assign (name ++ "[" ++ show idx ++ "]") (SomeExp value)

-- ============================================================================
-- Typed Matrix Views (HOAS-style, type-safe row/column indexing)
-- ============================================================================

-- | Type-safe row index wrapper (prevents swapping with column index)
newtype RowIndex = Row (Exp I32)

-- | Type-safe column index wrapper (prevents swapping with row index)
newtype ColIndex = Col (Exp I32)

-- | Matrix view with typed indices
-- This prevents accidentally swapping row and column indices
data MatrixView space n a = MatrixView
  { mvPtr    :: Ptr space (Array n a)
  , mvRows   :: Int
  , mvCols   :: Int
  , mvStride :: Int  -- Row stride (usually = cols for row-major)
  }

-- | Create a typed matrix view
-- Usage: matView <- makeMatrixView ptrA rows cols
makeMatrixView :: Ptr space (Array n a) -> Int -> Int -> MatrixView space n a
makeMatrixView ptr rows cols = MatrixView ptr rows cols cols

-- | Safe accessor with typed indices - PREVENTS row/column swapping
-- Usage: value <- matView `at` (Row i, Col j)
at :: MatrixView Storage n a -> (RowIndex, ColIndex) -> ShaderM (Exp a)
at (MatrixView ptr _ _ stride) (Row row, Col col) = do
  let offset = row `Mul` LitI32 stride `Add` col
  return $ PtrIndex ptr offset

-- | Safe write with typed indices
-- Usage: matView `writeAt` (Row i, Col j) $ value
writeAt :: MatrixView Storage n a -> (RowIndex, ColIndex) -> Exp a -> ShaderM ()
writeAt (MatrixView (Ptr name) _ _ stride) (Row row, Col col) value = do
  let offset = row `Mul` LitI32 stride `Add` col
  emitStmt $ Assign (name ++ "[" ++ show offset ++ "]") (SomeExp value)

-- ============================================================================
-- Subgroup Matrix Operations
-- ============================================================================

-- | Load data from buffer into subgroup matrix
subgroupMatrixLoad :: TypeRep           -- ^ Subgroup matrix type (TSubgroupMatrixLeft/Right)
                   -> Ptr s a           -- ^ Source buffer pointer
                   -> Exp U32           -- ^ Offset
                   -> Exp Bool_         -- ^ Transpose flag
                   -> Exp U32           -- ^ Stride
                   -> ShaderM (Exp b)   -- ^ Returns the subgroup matrix
subgroupMatrixLoad ty ptr offset transpose stride =
  return $ SubgroupMatrixLoad ty ptr offset transpose stride

-- | Multiply-accumulate for subgroup matrices
subgroupMatrixMultiplyAccumulate :: Exp a  -- ^ Left matrix (subgroup_matrix_left)
                                 -> Exp b  -- ^ Right matrix (subgroup_matrix_right)
                                 -> Exp c  -- ^ Accumulator (subgroup_matrix_result)
                                 -> ShaderM (Exp c)  -- ^ Returns updated accumulator
subgroupMatrixMultiplyAccumulate left right acc =
  return $ SubgroupMatrixMultiplyAccumulate left right acc

-- | Store subgroup matrix to buffer
subgroupMatrixStore :: Ptr s a     -- ^ Destination buffer pointer
                    -> Exp U32     -- ^ Offset
                    -> Exp b       -- ^ Source matrix (subgroup_matrix_result)
                    -> Exp Bool_   -- ^ Transpose flag
                    -> Exp U32     -- ^ Stride
                    -> ShaderM ()
subgroupMatrixStore ptr offset matrix transpose stride =
  emitStmt $ SubgroupMatrixStore ptr offset matrix transpose stride

-- | Compile-time loop for unrolling (equivalent to mapM_)
-- Distinguishes "Haskell-side loops (unrolling)" from "WGSL-side loops (runtime)"
staticFor :: Monad m => [a] -> (a -> m b) -> m ()
staticFor = flip mapM_

-- | Alias for staticFor with explicit void return
staticFor_ :: Monad m => [a] -> (a -> m b) -> m ()
staticFor_ = staticFor

-- | Float32 literal
litF :: Float -> Exp F32
litF = LitF32

-- | Int32 literal
litI :: Int -> Exp I32
litI = LitI32

-- | UInt32 literal
litU :: Int -> Exp U32
litU = LitU32

-- | Matrix role for subgroup operations
data MatrixRole = LeftMatrix | RightMatrix | ResultMatrix
  deriving (Eq, Show)

-- | Create a new subgroup matrix variable with automatic naming
-- Hides the TSubgroupMatrix... constructors from user code
newSubgroupMatrix :: MatrixRole -> TypeRep -> Int -> Int -> ShaderM (Ptr Private a)
newSubgroupMatrix role precision m n = do
  let ty = case role of
        LeftMatrix   -> TSubgroupMatrixLeft precision m n
        RightMatrix  -> TSubgroupMatrixRight precision m n
        ResultMatrix -> TSubgroupMatrixResult precision m n
  name <- freshVar $ case role of
        LeftMatrix   -> "left"
        RightMatrix  -> "right"
        ResultMatrix -> "acc"
  emitStmt $ DeclVar name ty Nothing
  return $ Ptr name

-- | Create a new subgroup matrix initialized to zero
newSubgroupMatrixZero :: MatrixRole -> TypeRep -> Int -> Int -> ShaderM (Ptr Private a)
newSubgroupMatrixZero role precision m n = do
  ptr <- newSubgroupMatrix role precision m n
  initializeSubgroupMatrix ptr role precision m n
  return ptr

-- | Initialize a subgroup matrix to zero
initializeSubgroupMatrix :: Ptr Private a -> MatrixRole -> TypeRep -> Int -> Int -> ShaderM ()
initializeSubgroupMatrix (Ptr name) role precision m n = do
  -- Generate WGSL: var_name = subgroup_matrix_xxx<precision, m, n>(0);
  let roleStr = case role of
        LeftMatrix   -> "subgroup_matrix_left"
        RightMatrix  -> "subgroup_matrix_right"
        ResultMatrix -> "subgroup_matrix_result"
      precisionStr = prettyTypeRep precision
      initCode = name ++ " = " ++ roleStr ++ "<" ++ precisionStr ++ ", " ++ show m ++ ", " ++ show n ++ ">(0);"
  emitStmt $ RawStmt initCode

-- | Load data from buffer into subgroup matrix (high-level wrapper)
loadMatrix :: Ptr Private a     -- ^ Destination matrix variable
           -> Ptr Storage b      -- ^ Source buffer
           -> Exp U32           -- ^ Offset
           -> Exp U32           -- ^ Stride
           -> TypeRep           -- ^ Matrix type (for code generation)
           -> ShaderM ()
loadMatrix dest src offset stride ty = do
  result <- subgroupMatrixLoad ty src offset (LitBool False) stride
  assign dest result

-- | Store subgroup matrix to buffer (high-level wrapper)
storeMatrix :: Ptr Storage a     -- ^ Destination buffer
            -> Exp U32           -- ^ Offset
            -> Ptr Private b     -- ^ Source matrix variable
            -> Exp U32           -- ^ Stride
            -> ShaderM ()
storeMatrix dest offset src stride = do
  srcVal <- readPtr src
  subgroupMatrixStore dest offset srcVal (LitBool False) stride

-- | Multiply-accumulate (high-level wrapper)
-- acc = acc + left * right
mma :: Ptr Private a -> Ptr Private b -> Ptr Private c -> ShaderM ()
mma acc left right = do
  accVal <- readPtr acc
  leftVal <- readPtr left
  rightVal <- readPtr right
  result <- subgroupMatrixMultiplyAccumulate leftVal rightVal accVal
  assign acc result

-- ============================================================================
-- Multi-dimensional Views (Safe Array Indexing)
-- ============================================================================

-- | Create a 1D view over an array buffer
-- This is mostly just for consistency - direct buffer access is fine for 1D
makeView1D :: Ptr space (Array n a) -> Int -> View space (View1D n a)
makeView1D ptr size = View1D ptr size

-- | Create a 2D view over a 1D array buffer
-- Treats the 1D buffer as a 2D matrix with row-major layout
-- @param rows - number of rows
-- @param cols - number of columns (also the row stride)
makeView2D :: Ptr space (Array n a) -> Int -> Int -> View space (View2D rows cols a)
makeView2D ptr rows cols = View2D ptr rows cols cols  -- stride = cols for row-major

-- | Create a 3D view over a 1D array buffer
-- @param d1, d2, d3 - dimensions
makeView3D :: Ptr space (Array n a) -> Int -> Int -> Int -> View space (View3D d1 d2 d3 a)
makeView3D ptr d1 d2 d3 = View3D ptr d1 d2 d3 (d2 P.* d3) d3

-- | Calculate 1D offset from 1D view index
viewIndex1D :: View space (View1D n a) -> Exp I32 -> Exp I32
viewIndex1D (View1D _ _) idx = idx

-- | Calculate 1D offset from 2D view indices (row, col)
-- offset = row * rowStride + col
viewIndex2D :: View space (View2D rows cols a) -> Exp I32 -> Exp I32 -> Exp I32
viewIndex2D (View2D _ _ _ stride) row col =
  row `Mul` LitI32 stride `Add` col

-- | Calculate 1D offset from 3D view indices (i, j, k)
-- offset = i * stride2 + j * stride3 + k
viewIndex3D :: View space (View3D d1 d2 d3 a) -> Exp I32 -> Exp I32 -> Exp I32 -> Exp I32
viewIndex3D (View3D _ _ _ _ stride2 stride3) i j k =
  i `Mul` LitI32 stride2 `Add` j `Mul` LitI32 stride3 `Add` k

-- | Read from a 1D view
readView1D :: View Storage (View1D n a) -> Exp I32 -> ShaderM (Exp a)
readView1D (View1D ptr _) idx = return $ PtrIndex ptr idx

-- | Read from a 2D view
readView2D :: View Storage (View2D rows cols a) -> Exp I32 -> Exp I32 -> ShaderM (Exp a)
readView2D view@(View2D ptr _ _ _) row col = do
  let offset = viewIndex2D view row col
  return $ PtrIndex ptr offset

-- | Read from a 3D view
readView3D :: View Storage (View3D d1 d2 d3 a) -> Exp I32 -> Exp I32 -> Exp I32 -> ShaderM (Exp a)
readView3D view@(View3D ptr _ _ _ _ _) i j k = do
  let offset = viewIndex3D view i j k
  return $ PtrIndex ptr offset

-- | Write to a 1D view
writeView1D :: View Storage (View1D n a) -> Exp I32 -> Exp a -> ShaderM ()
writeView1D (View1D (Ptr name) _) idx value =
  emitStmt $ Assign (name ++ "[" ++ show idx ++ "]") (SomeExp value)

-- | Write to a 2D view
writeView2D :: View Storage (View2D rows cols a) -> Exp I32 -> Exp I32 -> Exp a -> ShaderM ()
writeView2D view@(View2D (Ptr name) _ _ _) row col value = do
  let offset = viewIndex2D view row col
  emitStmt $ Assign (name ++ "[" ++ show offset ++ "]") (SomeExp value)

-- | Write to a 3D view
writeView3D :: View Storage (View3D d1 d2 d3 a) -> Exp I32 -> Exp I32 -> Exp I32 -> Exp a -> ShaderM ()
writeView3D view@(View3D (Ptr name) _ _ _ _ _) i j k value = do
  let offset = viewIndex3D view i j k
  emitStmt $ Assign (name ++ "[" ++ show offset ++ "]") (SomeExp value)

-- ============================================================================
-- Debug Print Support
-- ============================================================================

-- | Debug print for F32 values
-- Writes to a special debug buffer that can be read from the host
-- Note: Requires debug buffer to be set up (see buildKernelWithDebug)
debugPrintF :: String -> Exp F32 -> ShaderM ()
debugPrintF label value = do
  -- This is a simplified version - in practice, this would:
  -- 1. Use atomicAdd to get next index in debug buffer
  -- 2. Write a type tag
  -- 3. Write the value
  -- For now, we emit a comment as a placeholder
  emitStmt $ Comment $ "DEBUG: " ++ label ++ " = " ++ show value

-- | Debug print for I32 values
debugPrintI :: String -> Exp I32 -> ShaderM ()
debugPrintI label value = do
  emitStmt $ Comment $ "DEBUG: " ++ label ++ " = " ++ show value

-- | Debug print for U32 values
debugPrintU :: String -> Exp U32 -> ShaderM ()
debugPrintU label value = do
  emitStmt $ Comment $ "DEBUG: " ++ label ++ " = " ++ show value

-- | Generic debug print (delegates to type-specific version)
debugPrint :: String -> Exp a -> ShaderM ()
debugPrint label value = do
  emitStmt $ Comment $ "DEBUG: " ++ label ++ " = " ++ show value

-- ============================================================================
-- Automatic BindGroup Layout Management
-- ============================================================================

-- | Declare an input buffer with automatic binding assignment
-- The buffer is registered in order of declaration, and binding indices
-- are assigned sequentially (@binding(0), @binding(1), etc.)
declareInputBuffer :: String -> TypeRep -> ShaderM (Ptr Storage a)
declareInputBuffer name ty = declareStorageBuffer name ty MStorage

-- | Declare an output buffer with automatic binding assignment
declareOutputBuffer :: String -> TypeRep -> ShaderM (Ptr Storage a)
declareOutputBuffer name ty = declareStorageBuffer name ty MStorage

-- | Declare a storage buffer with automatic binding assignment
-- Registers the buffer in the shader state for auto-binding
declareStorageBuffer :: String -> TypeRep -> MemorySpace -> ShaderM (Ptr Storage a)
declareStorageBuffer name ty space = do
  -- Register this buffer in the declaration order
  modify $ \s -> s { declaredBuffers = declaredBuffers s ++ [(name, ty, space)] }
  -- Return a pointer to the buffer
  return $ Ptr name
