{-# LANGUAGE CPP #-}

-- | Safe GLFW bindings with automatic resource management
--
-- This module provides ContT-based resource management for GLFW resources.
-- All resources are automatically cleaned up when they go out of scope.
--
-- For unsafe low-level access, see 'Graphics.WebGPU.Dawn.GLFW.Internal'
-- (not recommended for normal use).
module Graphics.WebGPU.Dawn.GLFW
  ( module Graphics.WebGPU.Dawn.GLFW.ContT
  ) where

import Graphics.WebGPU.Dawn.GLFW.ContT
