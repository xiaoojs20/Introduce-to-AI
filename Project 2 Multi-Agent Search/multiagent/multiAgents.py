# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # print(f"currentGameState = \n{currentGameState}")
        # print(f"action = \n{action}")
        # print(f"successorGameState = \n{successorGameState}")
        # print(f"newPos = \n{newPos}")
        # print(f"newFood = \n{newFood}")
        # print(f"newGhostStates = \n{newGhostStates}")
        # print(f"newScaredTimes = \n{newScaredTimes}")
        # print(f"currentGameState.getScore() = \n{currentGameState.getScore()}")
        # print(f"successorGameState.getScore() = \n{successorGameState.getScore()}")
        "*** YOUR CODE HERE ***"
        newFoodList = newFood.asList()
        newFoodDist = [util.manhattanDistance(food, newPos) for food in newFoodList]
        if newFoodDist:
            minFoodDist = min(newFoodDist)
            foodScore = 1.0/minFoodDist
        else:
            foodScore = 0
        
        newGhostPos = successorGameState.getGhostPositions()
        newGhostDist = [util.manhattanDistance(ghost, newPos) for ghost in newGhostPos]

        allGhostDist = 0
        ghostScore = 0
        for i in range(len(newGhostDist)):
            if newScaredTimes[i] == 0:
                allGhostDist += newGhostDist[i]
            else: # scared
                ghostScore += 1.0/newGhostDist[i]
        if allGhostDist == 0:
            ghostScore += 1.0
        else:
            ghostScore += -1.0/allGhostDist

        # return successorGameState.getScore() + foodScore + ghostScore
        return successorGameState.getScore() + 2.0*foodScore + 3.0*ghostScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"  
        def terminalTest(gameState, depth):
            return gameState.isWin() or gameState.isLose() or depth == self.depth

        # for Pacman: agent 0
        def maxValue(gameState, depth, agentIndex=0):
            if terminalTest(gameState, depth):
                return self.evaluationFunction(gameState)
            v = -float('inf')
            legalMoves = gameState.getLegalActions(agentIndex)
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                v = max(v, minValue(successorGameState, depth, 1))
            return v

        # for Ghost: agent 1~(n-1)
        def minValue(gameState, depth, agentIndex):
            if terminalTest(gameState, depth):
                return self.evaluationFunction(gameState)
            v = float('inf')
            legalMoves = gameState.getLegalActions(agentIndex)
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents()-1:
                    # Pacman and Ghost搜索结束，进入下一个depth
                    v = min(v, maxValue(successorGameState, depth + 1, 0))
                else:
                    v = min(v, minValue(successorGameState, depth, agentIndex+1))
            return v
        
        actions = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in actions]
        minimaxValue = [minValue(successor, 0, 1) for successor in successors]
        maxIndex = minimaxValue.index(max(minimaxValue))
        return actions[maxIndex]
        util.raiseNotDefined()
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def terminalTest(gameState, depth):
            return gameState.isWin() or gameState.isLose() or depth == self.depth

        # for Pacman: agent 0
        def maxValue(gameState, depth, agentIndex, alpha, beta):
            a = Directions.STOP
            if terminalTest(gameState, depth):
                return self.evaluationFunction(gameState), a
            v = -float('inf')
            legalMoves = gameState.getLegalActions(agentIndex)
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                minV = minValue(successorGameState, depth, 1, alpha, beta)
                if v < minV:
                    v = minV
                    a = action
                if v > beta:
                    return v, a
                alpha = max(alpha, v)
            return v, a

        # for Ghost: agent 1~(n-1)
        def minValue(gameState, depth, agentIndex, alpha, beta):
            if terminalTest(gameState, depth):
                return self.evaluationFunction(gameState)
            v = float('inf')
            legalMoves = gameState.getLegalActions(agentIndex)
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents()-1:
                    # Pacman and Ghost搜索结束，进入下一个depth
                    v = min(v, maxValue(successorGameState, depth + 1, 0, alpha, beta)[0])
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                else:
                    v = min(v, minValue(successorGameState, depth, agentIndex+1, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
            return v
        
        _action = Directions.STOP
        # find the action with maxV
        maxV, maxAction = maxValue(gameState, 0, 0, -float('inf'), float('inf'))
        return maxAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Based on Minimax
        def terminalTest(gameState, depth):
            return gameState.isWin() or gameState.isLose() or depth == self.depth

        # for Pacman: agent 0
        def maxValue(gameState, depth, agentIndex=0):
            if terminalTest(gameState, depth):
                return self.evaluationFunction(gameState)
            v = -float('inf')
            legalMoves = gameState.getLegalActions(agentIndex)
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                v = max(v, sumPrValue(successorGameState, depth, 1))
            return v

        # for Ghost: agent 1~(n-1)
        def sumPrValue(gameState, depth, agentIndex):
            if terminalTest(gameState, depth):
                return self.evaluationFunction(gameState)
            v = 0
            legalMoves = gameState.getLegalActions(agentIndex)
            countChild = 0
            for action in legalMoves:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents()-1:
                    # Pacman and Ghost搜索结束，进入下一个depth
                    v += maxValue(successorGameState, depth + 1, 0)/len(legalMoves)
                else:
                    v += sumPrValue(successorGameState, depth, agentIndex+1)/len(legalMoves)
            return v
        
        actions = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in actions]
        minimaxValue = [sumPrValue(successor, 0, 1) for successor in successors]
        maxIndex = minimaxValue.index(max(minimaxValue))
        return actions[maxIndex]

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    
    currentFoodList = currentFood.asList()
    currentFoodDist = [util.manhattanDistance(food, currentPos) for food in currentFoodList]
    if currentFoodDist:
        minFoodDist = min(currentFoodDist)
        foodScore = 1.0/minFoodDist
    else:
        foodScore = 0
    
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    currentGhostPos = currentGameState.getGhostPositions()
    currentGhostDist = [util.manhattanDistance(ghost, currentPos) for ghost in currentGhostPos]

    allGhostDist = 0
    ghostScore = 0
    for i in range(len(currentGhostDist)):
        if currentScaredTimes[i] == 0:
            allGhostDist += currentGhostDist[i]
        else: # scared
            ghostScore += 1.0/currentGhostDist[i]
    if allGhostDist == 0:
        ghostScore += 1.0
    else:
        ghostScore += -1.0/allGhostDist

    return currentGameState.getScore() + 10*foodScore + 10*ghostScore
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
