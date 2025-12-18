{-# LANGUAGE CPP #-}

module Main where

#ifdef ENABLE_GLFW

import Control.Concurrent (threadDelay)
import Control.Monad (unless, when)
import Control.Monad.IO.Class (liftIO)
import Control.Monad.Trans.Cont (evalContT)
import Data.List (foldl', intercalate, find)
import Data.Maybe (isNothing, isJust, mapMaybe)
import Data.Time.Clock (UTCTime, diffUTCTime, getCurrentTime)
import qualified Data.Vector as V
import Graphics.WebGPU.Dawn.ContT
import Graphics.WebGPU.Dawn.GLFW
import Numeric (showFFloat)
import System.Random (StdGen, randomR, initStdGen)

type Point = (Int, Int)
type Color = (Float, Float, Float)

translate :: Point -> (Int, Int) -> Point
translate (x, y) (dx, dy) = (x + dx, y + dy)

data TetrominoKind = I | O | T | S | Z | J | L
  deriving (Eq, Enum, Bounded, Show)

data Tetromino = Tetromino
  { tetKind :: TetrominoKind
  , tetRotation :: Int
  , tetPosition :: Point
  } deriving (Show)

type Board = V.Vector (V.Vector (Maybe TetrominoKind))

data GameState = GameState
  { gameBoard :: Board
  , currentPiece :: Tetromino
  , upcomingPiece :: TetrominoKind
  , gameRng :: StdGen
  , dropTimer :: Double
  , clearedLines :: Int
  , scoreTotal :: Int
  , gameOver :: Bool
  }

data Square = Square
  { squareX :: Int
  , squareY :: Int
  , squareColor :: Color
  }

data VertexData = VertexData
  { vertexPos :: (Float, Float)
  , vertexColor :: Color
  }

data FrameGeometry = FrameGeometry
  { frameShader :: String
  , frameVertexCount :: Int
  }

data InputFrame = InputFrame
  { leftHeld :: Bool
  , rightHeld :: Bool
  , downHeld :: Bool
  , rotateCwHeld :: Bool
  , rotateCcwHeld :: Bool
  , hardDropHeld :: Bool
  } deriving (Eq)

data InputEvents = InputEvents
  { leftPressed :: Bool
  , rightPressed :: Bool
  , downPressed :: Bool
  , rotateCwPressed :: Bool
  , rotateCcwPressed :: Bool
  , hardDropPressed :: Bool
  , downIsHeld :: Bool
  }

windowWidth, windowHeight :: Int
windowWidth = 540
windowHeight = 800

boardWidth, boardHeight :: Int
boardWidth = 10
boardHeight = 20

cellWidth, cellHeight :: Float
cellWidth = 2.0 / fromIntegral boardWidth
cellHeight = 2.0 / fromIntegral boardHeight

spawnPosition :: Point
spawnPosition = (boardWidth `div` 2 - 2, -1)

baseDropInterval :: GameState -> Double
baseDropInterval st =
  let lvl = clearedLines st `div` 10
  in max 0.1 (0.7 - fromIntegral lvl * 0.05)

fastDropInterval :: GameState -> Double
fastDropInterval st = max 0.03 (baseDropInterval st / 8)

main :: IO ()
main = evalContT $ do
  liftIO $ putStrLn "Launching WebGPU Tetris (keys: arrows + Z/X/Space)..."
  liftIO glfwInit

  window <- createWindow windowWidth windowHeight "WebGPU Tetris"
  ctx <- createContext
  surface <- createSurfaceForWindow ctx window
  format <- liftIO $ getSurfacePreferredFormat surface
  liftIO $ configureSurface surface windowWidth windowHeight

  liftIO $ do
    gen <- initStdGen
    startState <- pure $ initialGameState gen
    startTime <- getCurrentTime
    loopGame ctx window surface format startState emptyInputFrame startTime

  liftIO glfwTerminate
  liftIO $ putStrLn "Goodbye Tetris!"

loopGame :: Context -> Window -> Surface -> Int -> GameState -> InputFrame -> UTCTime -> IO ()
loopGame ctx window surface format state prevInput lastTick = do
  pollEvents
  shouldClose <- windowShouldClose window
  unless shouldClose $ do
    now <- getCurrentTime
    currentInput <- readInput window
    let events = deriveEvents prevInput currentInput
        dt = realToFrac (diffUTCTime now lastTick)
        nextState =
          if gameOver state
            then state
            else updateGame dt events state
        frame = frameGeometry nextState
    renderFrame ctx surface format frame
    when (gameOver nextState && not (gameOver state)) $
      putStrLn "Game over! Close the window to exit."
    threadDelay 16000
    loopGame ctx window surface format nextState currentInput now

renderFrame :: Context -> Surface -> Int -> Maybe FrameGeometry -> IO ()
renderFrame ctx surface format geometry = do
  evalContT $ do
    texture <- createCurrentTexture surface
    view <- createTextureView texture
    encoder <- createCommandEncoder ctx
    pass <- createRenderPass encoder view

    case geometry of
      Just (FrameGeometry shaderSrc vertexCount) -> do
        shader <- createShaderModule ctx shaderSrc
        pipeline <- createRenderPipeline ctx shader format
        liftIO $ do
          setRenderPipeline pass pipeline
          draw pass vertexCount
      Nothing ->
        liftIO $ pure ()

    liftIO $ endRenderPass pass
    commands <- createCommandBuffer encoder
    liftIO $ submitCommand ctx commands
  surfacePresent surface

initialGameState :: StdGen -> GameState
initialGameState gen0 =
  let (firstKind, gen1) = randomKind gen0
      (nextKind, gen2) = randomKind gen1
      piece = Tetromino firstKind 0 spawnPosition
  in GameState
      { gameBoard = emptyBoard
      , currentPiece = piece
      , upcomingPiece = nextKind
      , gameRng = gen2
      , dropTimer = 0
      , clearedLines = 0
      , scoreTotal = 0
      , gameOver = False
      }

updateGame :: Double -> InputEvents -> GameState -> GameState
updateGame dt inputs state =
  let afterInput = handleInput inputs state
      (afterFall, timer') = stepGravity dt (downIsHeld inputs) afterInput
  in afterFall { dropTimer = timer' }

handleInput :: InputEvents -> GameState -> GameState
handleInput inputs state
  | hardDropPressed inputs = hardDrop state
  | otherwise =
      let movedLeft = if leftPressed inputs then moveHorizontal (-1) state else state
          movedRight = if rightPressed inputs then moveHorizontal 1 movedLeft else movedLeft
          rotatedCw = if rotateCwPressed inputs then rotatePieceCW movedRight else movedRight
          rotatedCcw = if rotateCcwPressed inputs then rotatePieceCCW rotatedCw else rotatedCw
          softDropped = if downPressed inputs then fst (dropStep rotatedCcw) else rotatedCcw
      in softDropped

stepGravity :: Double -> Bool -> GameState -> (GameState, Double)
stepGravity dt accelerate state = advance (dropTimer state + dt) state
  where
    interval = if accelerate then fastDropInterval state else baseDropInterval state
    advance t s
      | t < interval = (s, t)
      | otherwise =
          let (s', locked) = dropStep s
              newTimer = if locked then 0 else t - interval
          in advance newTimer s'

dropStep :: GameState -> (GameState, Bool)
dropStep state =
  case tryMove (0, 1) state of
    Just moved -> (state { currentPiece = moved }, False)
    Nothing    -> (lockPiece state, True)

hardDrop :: GameState -> GameState
hardDrop state =
  let descend piece =
        case movePiece (gameBoard state) (0, 1) piece of
          Just nextPiece -> descend nextPiece
          Nothing -> piece
      finalPiece = descend (currentPiece state)
  in lockPiece state { currentPiece = finalPiece }

moveHorizontal :: Int -> GameState -> GameState
moveHorizontal dx state =
  case tryMove (dx, 0) state of
    Just moved -> state { currentPiece = moved }
    Nothing    -> state

rotatePieceCW :: GameState -> GameState
rotatePieceCW = rotatePieceWith (\r -> (r + 1) `mod` 4)

rotatePieceCCW :: GameState -> GameState
rotatePieceCCW = rotatePieceWith (\r -> (r + 3) `mod` 4)

rotatePieceWith :: (Int -> Int) -> GameState -> GameState
rotatePieceWith nextRotation state =
  case tryRotate nextRotation state of
    Just rotated -> state { currentPiece = rotated }
    Nothing      -> state

tryRotate :: (Int -> Int) -> GameState -> Maybe Tetromino
tryRotate nextRot GameState { gameBoard = board, currentPiece = piece } =
  let rotated = piece { tetRotation = nextRot (tetRotation piece) }
      kicks = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0)]
      candidates = [ rotated { tetPosition = translate (tetPosition rotated) shift } | shift <- kicks ]
  in find (validPosition board) candidates

tryMove :: (Int, Int) -> GameState -> Maybe Tetromino
tryMove (dx, dy) GameState { gameBoard = board, currentPiece = piece } =
  movePiece board (dx, dy) piece

lockPiece :: GameState -> GameState
lockPiece state =
  let boardWithPiece = placePiece (gameBoard state) (currentPiece state)
      (cleansed, cleared) = clearLines boardWithPiece
      (nextKind, gen') = randomKind (gameRng state)
      newPiece = Tetromino (upcomingPiece state) 0 spawnPosition
      updatedState = state
        { gameBoard = cleansed
        , clearedLines = clearedLines state + cleared
        , scoreTotal = scoreTotal state + scored cleared
        , dropTimer = 0
        , upcomingPiece = nextKind
        , gameRng = gen'
        , currentPiece = newPiece
        }
      gameOverNow = not (validPosition cleansed newPiece)
  in updatedState { gameOver = gameOverNow }

scored :: Int -> Int
scored linesCleared =
  case linesCleared of
    1 -> 40
    2 -> 100
    3 -> 300
    4 -> 1200
    _ -> 0

frameGeometry :: GameState -> Maybe FrameGeometry
frameGeometry state =
  let squares = settledSquares (gameBoard state) ++ activeSquares (currentPiece state)
      vertices = squaresToVertices squares
  in if null vertices
       then Nothing
       else Just FrameGeometry
              { frameShader = shaderFromVertices vertices
              , frameVertexCount = length vertices
              }

settledSquares :: Board -> [Square]
settledSquares board =
  concat $
    V.toList $
      V.imap (\y row ->
        V.ifoldl' (\acc x cell -> maybe acc (\kind -> Square x y (pieceColor kind) : acc) cell) [] row
      ) board

activeSquares :: Tetromino -> [Square]
activeSquares piece =
  mapMaybe toSquare (tetrominoBlocks piece)
  where
    color = pieceColor (tetKind piece)
    toSquare (x, y)
      | y < 0 || y >= boardHeight = Nothing
      | x < 0 || x >= boardWidth = Nothing
      | otherwise = Just $ Square x y color

squaresToVertices :: [Square] -> [VertexData]
squaresToVertices = concatMap squareVertices

squareVertices :: Square -> [VertexData]
squareVertices Square { squareX = x, squareY = y, squareColor = col } =
  let x0 = -1.0 + fromIntegral x * cellWidth
      x1 = x0 + cellWidth
      y0 = 1.0 - fromIntegral y * cellHeight
      y1 = y0 - cellHeight
      p0 = (x0, y0)
      p1 = (x0, y1)
      p2 = (x1, y1)
      p3 = (x1, y0)
  in [ VertexData p0 col
     , VertexData p1 col
     , VertexData p2 col
     , VertexData p0 col
     , VertexData p2 col
     , VertexData p3 col
     ]

shaderFromVertices :: [VertexData] -> String
shaderFromVertices verts =
  let vertexCount = length verts
      positions = intercalate ",\n    " $ map (formatVec2 . vertexPos) verts
      colors = intercalate ",\n    " $ map (formatVec3 . vertexColor) verts
  in unlines
      [ "struct VertexOutput {"
      , "  @builtin(position) position : vec4f,"
      , "  @location(0) color : vec3f,"
      , "};"
      , ""
      , "const POSITIONS : array<vec2f," ++ show vertexCount ++ "> = array<vec2f," ++ show vertexCount ++ ">("
      , "    " ++ positions
      , ");"
      , ""
      , "const COLORS : array<vec3f," ++ show vertexCount ++ "> = array<vec3f," ++ show vertexCount ++ ">("
      , "    " ++ colors
      , ");"
      , ""
      , "@vertex"
      , "fn vertexMain(@builtin(vertex_index) idx : u32) -> VertexOutput {"
      , "  var out : VertexOutput;"
      , "  out.position = vec4f(POSITIONS[idx], 0.0, 1.0);"
      , "  out.color = COLORS[idx];"
      , "  return out;"
      , "}"
      , ""
      , "@fragment"
      , "fn fragmentMain(input : VertexOutput) -> @location(0) vec4f {"
      , "  return vec4f(input.color, 1.0);"
      , "}"
      ]

formatVec2 :: (Float, Float) -> String
formatVec2 (x, y) = "vec2f(" ++ fmt x ++ ", " ++ fmt y ++ ")"

formatVec3 :: Color -> String
formatVec3 (r, g, b) = "vec3f(" ++ fmt r ++ ", " ++ fmt g ++ ", " ++ fmt b ++ ")"

fmt :: Float -> String
fmt val =
  let rendered = showFFloat (Just 4) val ""
  in if '.' `elem` rendered then rendered else rendered ++ ".0"

emptyBoard :: Board
emptyBoard = V.replicate boardHeight (V.replicate boardWidth Nothing)

pieceColor :: TetrominoKind -> Color
pieceColor kind =
  case kind of
    I -> (0.0, 0.8, 0.9)
    O -> (0.95, 0.85, 0.15)
    T -> (0.75, 0.25, 0.85)
    S -> (0.2, 0.8, 0.2)
    Z -> (0.9, 0.2, 0.2)
    J -> (0.2, 0.35, 0.95)
    L -> (0.95, 0.55, 0.2)

tetrominoBlocks :: Tetromino -> [Point]
tetrominoBlocks Tetromino { tetKind = kind, tetRotation = rot, tetPosition = (px, py) } =
  [ (px + ox, py + oy) | (ox, oy) <- rotationTable kind !! (rot `mod` 4) ]

rotationTable :: TetrominoKind -> [[Point]]
rotationTable kind =
  take 4 $ iterate (map rotatePoint) (baseShape kind)
  where
    rotatePoint (x, y) = (3 - y, x)

baseShape :: TetrominoKind -> [Point]
baseShape kind =
  case kind of
    I -> [(0, 1), (1, 1), (2, 1), (3, 1)]
    O -> [(1, 1), (1, 2), (2, 1), (2, 2)]
    T -> [(1, 0), (0, 1), (1, 1), (2, 1)]
    S -> [(1, 1), (2, 1), (0, 2), (1, 2)]
    Z -> [(0, 1), (1, 1), (1, 2), (2, 2)]
    J -> [(0, 0), (0, 1), (1, 1), (2, 1)]
    L -> [(2, 0), (0, 1), (1, 1), (2, 1)]

placePiece :: Board -> Tetromino -> Board
placePiece board piece =
  foldl' (\acc (x, y) -> place x y acc) board (tetrominoBlocks piece)
  where
    color = tetKind piece
    place x y acc
      | y < 0 || y >= boardHeight = acc
      | x < 0 || x >= boardWidth = acc
      | otherwise =
          let row = acc V.! y
              row' = row V.// [(x, Just color)]
          in acc V.// [(y, row')]

clearLines :: Board -> (Board, Int)
clearLines board =
  let rows = V.toList board
      (filled, remaining) = foldr classify ([], []) rows
      classify row (cleared, rest)
        | rowFull row = (row : cleared, rest)
        | otherwise   = (cleared, row : rest)
      clearedCount = length filled
      newRows = replicate clearedCount emptyRow ++ remaining
  in (V.fromList newRows, clearedCount)
  where
    rowFull = V.all isJust
    emptyRow = V.replicate boardWidth Nothing

validPosition :: Board -> Tetromino -> Bool
validPosition board piece =
  all (cellFree board) (tetrominoBlocks piece)

cellFree :: Board -> Point -> Bool
cellFree _ (_, y) | y < 0 = True
cellFree _ (_, y) | y >= boardHeight = False
cellFree _ (x, _) | x < 0 || x >= boardWidth = False
cellFree board (x, y) =
  isNothing $ (board V.! y) V.! x

movePiece :: Board -> (Int, Int) -> Tetromino -> Maybe Tetromino
movePiece board (dx, dy) piece =
  let (px, py) = tetPosition piece
      moved = piece { tetPosition = (px + dx, py + dy) }
  in if validPosition board moved then Just moved else Nothing

randomKind :: StdGen -> (TetrominoKind, StdGen)
randomKind gen =
  let (idx, gen') = randomR (fromEnum (minBound :: TetrominoKind), fromEnum (maxBound :: TetrominoKind)) gen
  in (toEnum idx, gen')

readInput :: Window -> IO InputFrame
readInput window = do
  let pressed key = (/= 0) <$> windowGetKey window key
  l <- pressed keyLeft
  r <- pressed keyRight
  d <- pressed keyDown
  z <- pressed keyZ
  x <- pressed keyX
  space <- pressed keySpace
  up <- pressed keyUp
  pure InputFrame
    { leftHeld = l
    , rightHeld = r
    , downHeld = d
    , rotateCwHeld = x || up
    , rotateCcwHeld = z
    , hardDropHeld = space
    }

deriveEvents :: InputFrame -> InputFrame -> InputEvents
deriveEvents prev current = InputEvents
  { leftPressed = transitioned leftHeld
  , rightPressed = transitioned rightHeld
  , downPressed = transitioned downHeld
  , rotateCwPressed = transitioned rotateCwHeld
  , rotateCcwPressed = transitioned rotateCcwHeld
  , hardDropPressed = transitioned hardDropHeld
  , downIsHeld = downHeld current
  }
  where
    transitioned field = field current && not (field prev)

emptyInputFrame :: InputFrame
emptyInputFrame = InputFrame False False False False False False

keyLeft, keyRight, keyDown, keyUp, keyZ, keyX, keySpace :: Int
keyLeft = 263
keyRight = 262
keyDown = 264
keyUp = 265
keyZ = 90
keyX = 88
keySpace = 32

#else

main :: IO ()
main = putStrLn "This example requires GLFW support. Build with -fglfw."

#endif
