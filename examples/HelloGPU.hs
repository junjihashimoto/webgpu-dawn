-- | Simple example demonstrating WebGPU Dawn usage with ContT
module Main where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V

main :: IO ()
main = evalContT $ do
  liftIO $ putStrLn "Creating WebGPU context..."
  ctx <- createContext
  liftIO $ putStrLn "Context created successfully!"

  liftIO $ putStrLn "\nCreating a simple tensor..."
  let shape = Shape [4]
      input = V.fromList [1.0, 2.0, 3.0, 4.0 :: Float]

  tensor <- createTensorWithData ctx shape input
  liftIO $ putStrLn "Tensor created!"

  liftIO $ putStrLn "\nReading data back from GPU..."
  output <- liftIO $ fromGPU ctx tensor 4 :: ContT r IO (V.Vector Float)

  liftIO $ putStrLn $ "Input:  " ++ show input
  liftIO $ putStrLn $ "Output: " ++ show output

  liftIO $ putStrLn "\nSuccess! WebGPU Dawn is working."
  -- All resources automatically cleaned up here!
