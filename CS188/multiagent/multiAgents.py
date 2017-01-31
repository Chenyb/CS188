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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        "*** YOUR CODE HERE ***"
        score =  float("inf")
        pacmanPos = successorGameState.getPacmanPosition()
        foodList = currentGameState.getFood().asList()
        if foodList == []:
            return score

        if action == 'Stop':
            return -float("inf")

        for ghostState in newGhostStates:
            if ghostState.getPosition() == tuple(pacmanPos) and ghostState.scaredTimer is 0:
                return -float("inf") 
        #Using manhattanDistance to sort PacmanPosition
        for food in foodList:
            # the smaller manhattanDis is, the higher score 
            score = min(util.manhattanDistance(food,pacmanPos), score) 

        return -score

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
        numOfAgents = gameState.getNumAgents()
        depth = self.depth
        
        score = -float("inf")
        value = score
        actions = gameState.getLegalActions(0)

        def minMaxValue(state, index, level):
            if (index % numOfAgents == 0):
              return maxValue(state, index, level)
            else:
              return minValue(state, index, level)
 
        def maxValue(state, index, level):
            if level >= depth * numOfAgents-1 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            value = -float("inf")
            actions = state.getLegalActions(index % numOfAgents)
            for action in actions:
                newState = state.generateSuccessor(index % numOfAgents, action)
                value = max(value, minMaxValue(newState, index+1, level+1))
            return value
 
        def minValue(state, index, level):
            if level >= depth * numOfAgents-1 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            value = float("inf")
            actions = state.getLegalActions(index % numOfAgents)
            for action in actions:
                newState = state.generateSuccessor(index % numOfAgents, action)
                value = min(value, minMaxValue(newState, index+1, level+1))
            return value

        for action in actions:
            newState = gameState.generateSuccessor(0, action)
            value = max(value, minMaxValue(newState, 1, 0))
            if value > score:
                score = value
                resultAction = action
        return resultAction
 
        


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numOfAgents = gameState.getNumAgents()
        score = -float("inf")
        alpha = -float("inf")
        beta = float("inf")
        value = score
 
        def minMaxValue(state, index, level, alpha, beta):
            if index % numOfAgents == 0:
                return maxValue(state, index, level, alpha, beta)
            else:
                return minValue(state, index, level, alpha, beta)
 
        def maxValue(state, index, level, alpha, beta):
            if level >= self.depth * numOfAgents-1 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            value = -float("inf")
            for action in state.getLegalActions(index % numOfAgents):
                newState = state.generateSuccessor(index % numOfAgents, action)
                value = max(value, minMaxValue(newState, index+1, level+1, alpha, beta))
                if value > beta:
                    return value
                alpha = max(alpha, value)
            return value
 
        def minValue(state, index, level, alpha, beta):
            if level >= self.depth * numOfAgents-1 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            value = float("inf")
            for action in state.getLegalActions(index % numOfAgents):
                newState = state.generateSuccessor(index % numOfAgents, action)
                value = min(value, minMaxValue(newState, index+1, level+1, alpha, beta))
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value
 

        for action in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0, action)
            value = max(value, minMaxValue(newState, 1, 0, alpha, beta))
            if value > score:
                score = value
                resultAction = action
            alpha = max(alpha, value)
        return resultAction

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
        value = float(0)
        score = -float("inf")
        numOfAgents = gameState.getNumAgents()


        def minMaxValue(state, index, level):
            if index % numOfAgents == 0:
                return maxValue(state, index, level)
            else:
                return expValue(state, index, level)
 
        def maxValue(state, index, level):
            if level >= self.depth * numOfAgents-1 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            value = 0
            for action in state.getLegalActions(index % numOfAgents):
                newState = state.generateSuccessor(index % numOfAgents, action)
                value = max(value, minMaxValue(newState, index+1, level+1))
            return value
 
        def expValue(state, index, level):
            if level >= self.depth * numOfAgents-1 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            value = 0
            for action in state.getLegalActions(index % numOfAgents):
                newState = state.generateSuccessor(index % numOfAgents, action)
                value += float(minMaxValue(newState, index+1, level+1))
            return float(value)/float(len(state.getLegalActions(index % numOfAgents)))

        for action in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0, action)
            value = max(value, minMaxValue(newState, 1, 0))
            if value > score:
                score = value
                resultAction = action
        return resultAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    # 
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose(): 
        return -float('inf')
    elif currentGameState.isWin():
        return float('inf')
        
    foodList = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    pacmanPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    currentManhattan = float('inf')
    foodManhattanDistance = 0
    scareManhattan = 0
    result = 0 
    if foodList == []:
        return float('inf')
    for ghostState in ghostStates:
        currentManhattan = min(util.manhattanDistance(ghostState.getPosition(),pacmanPos), currentManhattan)
        if currentManhattan<2:
            return -float('inf')
        if (ghostState.scaredTimer > 0):
            scareManhattan = currentManhattan
            currentManhattan = 0
    for food in foodList:
        # the smaller manhattanDis is, the higher score 
        foodManhattanDistance = min(util.manhattanDistance(food,pacmanPos), foodManhattanDistance) 
    result +=  currentGameState.getScore()*4 - 4 * foodManhattanDistance
    result +=  1.0/max(currentManhattan,4) - scareManhattan - 100.0*len(capsules)
    return result

# Abbreviation
better = betterEvaluationFunction

