-- | Simple example demonstrating WebGPU Dawn usage
module Main where

import Graphics.WebGPU.Dawn
import qualified Data.Vector.Storable as V

main :: IO ()
main = do
  putStrLn "Creating WebGPU context..."
  ctx <- createContext
  putStrLn "Context created successfully!"

  putStrLn "\nCreating a simple tensor..."
  let shape = Shape [4]
      input = V.fromList [1.0, 2.0, 3.0, 4.0 :: Float]

  tensor <- createTensorWithData ctx shape input
  putStrLn "Tensor created!"

  putStrLn "\nReading data back from GPU..."
  output <- fromGPU ctx tensor 4 :: IO (V.Vector Float)

  putStrLn $ "Input:  " ++ show input
  putStrLn $ "Output: " ++ show output

  putStrLn "\nSuccess! WebGPU Dawn is working."
