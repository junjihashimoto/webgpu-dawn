module Main where

import Test.Hspec

main :: IO ()
main = hspec $ do
  describe "webgpu-dawn" $ do
    it "placeholder test" $ do
      True `shouldBe` True
