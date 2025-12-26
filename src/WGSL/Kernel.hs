{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE FlexibleInstances #-}

{-|
Module: WGSL.Kernel
Description: Composable kernel abstraction for kernel fusion

This module provides a high-level abstraction for composable GPU kernels.
Instead of writing imperative procedures (ShaderM ()), we treat kernels as
composable functions (Input -> Output), enabling kernel fusion optimizations.

Key benefits:
  - Compose operations with Category instance (f >>> g)
  - Fuse multiple operations into single shader pass
  - Reduce global memory traffic
  - Type-safe workgroup size tracking

Example:
@
  -- Fuse: Load -> Multiply -> Add -> ReLU -> Store into single pass
  fusedKernel :: Kernel 256 1 1 (Exp I32) ()
  fusedKernel =
    loadVec inputPtr
    >>> mapK (* 2.0)
    >>> mapK (+ 1.0)
    >>> mapK relu
    >>> storeVec outputPtr
@
-}

module WGSL.Kernel
  ( -- * Core Types
    Kernel(..)

    -- * Composition
  , mapK

    -- * Re-exports
  , (>>>)
  , Category(..)
  ) where

import Prelude hiding (id, (.))
import Control.Category
import GHC.TypeLits (Nat)
import WGSL.Monad (ShaderM)
import WGSL.AST (Exp)

-- | A composable kernel function running on a specific workgroup size.
--
-- Type parameters:
--   wX, wY, wZ: Workgroup dimensions (type-level Nats)
--   i: Input type (e.g., Exp F32, or a View)
--   o: Output type
--
-- This abstraction allows us to compose operations and perform kernel fusion.
-- Multiple operations can be fused into a single shader pass, reducing
-- global memory roundtrips.
newtype Kernel (wX :: Nat) (wY :: Nat) (wZ :: Nat) i o =
  Kernel { unKernel :: i -> ShaderM o }

-- | Category instance for Kernel composition
--
-- This allows us to compose kernels using (>>>) from Control.Category:
--   f >>> g  -- f runs first, passes result to g
--
-- Example:
--   load >>> process >>> store
--
-- The composition is performed in the ShaderM monad, so side effects
-- (like memory operations) are properly sequenced.
instance Category (Kernel wX wY wZ) where
  -- Identity kernel: passes input through unchanged
  id = Kernel return

  -- Composition: g . f means "f first, then g"
  -- Since we're in a monad, we use >>= to sequence the operations
  Kernel g . Kernel f = Kernel (\x -> f x >>= g)

-- | Lift a pure DSL function into a Kernel
--
-- This allows you to turn any pure expression transformation
-- (like (* 2.0) or (+ 1.0)) into a composable kernel.
--
-- Example:
--   mapK (* 2.0)  :: Kernel wX wY wZ (Exp F32) (Exp F32)
--   mapK (+ 1.0)  :: Kernel wX wY wZ (Exp F32) (Exp F32)
--   mapK relu     :: Kernel wX wY wZ (Exp F32) (Exp F32)
--
-- These can be composed:
--   mapK (* 2.0) >>> mapK (+ 1.0) >>> mapK relu
mapK :: (Exp a -> Exp b) -> Kernel wX wY wZ (Exp a) (Exp b)
mapK f = Kernel (return . f)
