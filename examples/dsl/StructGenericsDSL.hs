{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}

{-|
Struct Generation with GHC.Generics - DSL4 Feature Demo

This example demonstrates automatic WGSL struct generation from Haskell types
using GHC.Generics, as proposed in dsl4.md.

Features:
1. Automatic struct definition generation from Haskell records
2. Type-safe field access
3. No manual WGSL string manipulation

Compare this to manual struct definition - much cleaner!
-}

module Main where

import Prelude (Int, Float, IO, (++), show, putStrLn, ($), Maybe(..), mapM_)
import qualified Prelude as P
import GHC.Generics
import Data.Proxy

import WGSL.AST

-- | Define a Particle type in Haskell
-- This automatically generates a corresponding WGSL struct definition
data Particle = Particle
  { position :: Vec3 F32
  , velocity :: Vec3 F32
  , mass     :: F32
  } deriving Generic

-- | Derive WGSLType instance automatically using Generics
instance WGSLType Particle where
  wgslTypeRep _ = TStruct "Particle"

-- | Another example: a simpler VertexInput struct
data VertexInput = VertexInput
  { pos :: Vec3 F32
  , uv  :: Vec2 F32
  } deriving Generic

instance WGSLType VertexInput where
  wgslTypeRep _ = TStruct "VertexInput"

main :: IO ()
main = do
  putStrLn "=== WGSL Struct Generation with GHC.Generics ==="
  putStrLn ""
  putStrLn "This demonstrates automatic struct generation from Haskell types."
  putStrLn ""

  -- Generate struct definition for Particle
  putStrLn "1. Particle struct (position, velocity, mass):"
  putStrLn "   Haskell definition:"
  putStrLn "   data Particle = Particle"
  putStrLn "     { position :: Vec3 F32"
  putStrLn "     , velocity :: Vec3 F32"
  putStrLn "     , mass     :: F32"
  putStrLn "     } deriving (Generic)"
  putStrLn ""

  case wgslStructDef (Proxy :: Proxy Particle) of
    Just structDef -> do
      putStrLn "   Generated WGSL:"
      putStrLn P.$ "   " ++ show structDef
      putStrLn ""
      putStrLn P.$ "   WGSL struct name: " ++ wgslTypeName (Proxy :: Proxy Particle)
      putStrLn "   Fields:"
      mapM_ (\(fname, ftype) -> putStrLn P.$ "     " ++ fname ++ ": " ++ show ftype) (structFields structDef)
    Nothing ->
      putStrLn "   ERROR: No struct definition generated!"

  putStrLn ""
  putStrLn "2. VertexInput struct (pos, uv):"
  putStrLn "   Haskell definition:"
  putStrLn "   data VertexInput = VertexInput"
  putStrLn "     { pos :: Vec3 F32"
  putStrLn "     , uv  :: Vec2 F32"
  putStrLn "     } deriving (Generic)"
  putStrLn ""

  case wgslStructDef (Proxy :: Proxy VertexInput) of
    Just structDef -> do
      putStrLn "   Generated WGSL:"
      putStrLn P.$ "   " ++ show structDef
      putStrLn ""
      putStrLn P.$ "   WGSL struct name: " ++ wgslTypeName (Proxy :: Proxy VertexInput)
      putStrLn "   Fields:"
      mapM_ (\(fname, ftype) -> putStrLn P.$ "     " ++ fname ++ ": " ++ show ftype) (structFields structDef)
    Nothing ->
      putStrLn "   ERROR: No struct definition generated!"

  putStrLn ""
  putStrLn "Key Benefits:"
  putStrLn "  ✓ No manual WGSL string manipulation"
  putStrLn "  ✓ Type-safe struct definitions"
  putStrLn "  ✓ Automatic field name and type extraction"
  putStrLn "  ✓ Single source of truth (Haskell type)"
  putStrLn ""

  putStrLn "Next Steps (Future Work):"
  putStrLn "  • Field access DSL: particle.position, particle.velocity"
  putStrLn "  • Automatic buffer binding"
  putStrLn "  • CPU-GPU data marshaling"
