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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # return childGameState.getScore()

        if childGameState.isWin():
            return float("inf")

        # 计算ghost_evaluate
        ghost_evaluate = 0
        dist_with_normal_ghost = []
        dist_with_scared_ghost = []
        min_dist_with_normal_ghost = 1000
        min_dist_with_scared_ghost = 0
        for newGhostState in newGhostStates:
            if newGhostState.scaredTimer > 4 and util.manhattanDistance(newGhostState.getPosition(), newPos) < 2:
                return float("inf")
            if newGhostState.scaredTimer == 0:
                if util.manhattanDistance(newGhostState.getPosition(), newPos) < 2:
                    return float("-inf")
                dist_with_normal_ghost.append(util.manhattanDistance(newGhostState.getPosition(), newPos))
            else:
                dist_with_scared_ghost.append(util.manhattanDistance(newGhostState.getPosition(), newPos))
        if dist_with_normal_ghost:
            min_dist_with_normal_ghost = min(dist_with_normal_ghost)
        if dist_with_scared_ghost:
            min_dist_with_scared_ghost = min(dist_with_scared_ghost)
        # ghost_evaluate = -100 * min_dist_with_normal_ghost + min_dist_with_scared_ghost
        ghost_evaluate = -1 * min_dist_with_normal_ghost

        # 计算food_evaluate
        dist_with_food = []
        for food in list(newFood.asList()):
            dist_with_food.append(util.manhattanDistance(food, newPos))

        food_evaluate = -min(dist_with_food)
        if (currentGameState.getNumFood() > childGameState.getNumFood()):
            food_evaluate += 100
        if (currentGameState.getNumAgents() > childGameState.getNumAgents()):
            ghost_evaluate += 1000
        # print("childGameState.getScore():", childGameState.getScore())
        # print("food_evaluate:", food_evaluate)
        # print("ghost_evaluate:", ghost_evaluate)

        return childGameState.getScore() + food_evaluate + ghost_evaluate



def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        maxValue = float("-inf")
        maxAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            nextState = gameState.getNextState(0, action)
            nextValue = self.minimaxScore(nextState, 1, self.depth) # self.depth->0

            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action

        return maxAction

    def minimaxScore(self, state, agentIndex, depth):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if agentIndex == 0:  # pacman turn
            nextAgent = 1
            nextDepth = depth
        else:  # ghost turn
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            if nextAgent == 0:
                nextDepth = depth - 1
            else:
                nextDepth = depth

        legalActions = state.getLegalActions(agentIndex)
        successors = []
        for action in legalActions:
            successor = state.getNextState(agentIndex, action)
            successors.append(successor)
        scores = []
        for successor in successors:
            score = self.minimaxScore(successor, nextAgent, nextDepth)
            scores.append(score)
        if agentIndex == 0:
            return max(scores)
        else:
            return min(scores)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState: GameState):

        def AlphaBetaScore(gameState, currentDepth, agentIndex, alpha, beta):
            if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            maxValue = float("-inf")
            minValue = float("inf")

            if agentIndex == 0:  # pacman turn
                nextAgent = 1
                nextDepth = currentDepth
            else:  # ghost turn
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                if nextAgent == 0:
                    nextDepth = currentDepth + 1
                else:
                    nextDepth = currentDepth

            #   pac turn:   renew alpha=max(alpha, child_beta, child_alpha)
            if agentIndex == 0:
                for action in gameState.getLegalActions(agentIndex):
                    maxValue = max(maxValue, AlphaBetaScore(gameState.getNextState(agentIndex, action), nextDepth, nextAgent, alpha, beta))
                    if maxValue > beta:
                        return maxValue
                    alpha = max(alpha, maxValue)
                return maxValue
            #   ghosts turn: renew beta=min(beta, child_beta, child_alpha)
            else:
                for action in gameState.getLegalActions(agentIndex):
                    minValue = min(minValue, AlphaBetaScore(gameState.getNextState(agentIndex, action), nextDepth, nextAgent, alpha, beta))
                    if minValue < alpha:
                        return minValue
                    beta = min(beta, minValue)
                return minValue

        maxValue = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        maxAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            nextState = gameState.getNextState(0, action)

            nextValue = AlphaBetaScore(nextState, 0, 1, alpha, beta)

            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action

            alpha = max(alpha, maxValue)

        return maxAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        maxValue = float("-inf")
        maxAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            nextState = gameState.getNextState(0, action)

            nextValue = self.expectScore(nextState, 1, self.depth)

            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action

        return maxAction

    def get_average_Score(self, scores):
        # print(scores)
        # print( len(scores))
        average = sum(scores) / len(scores)
        return average
    def expectScore(self, state, agentIndex: int, depth: int):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if agentIndex == 0:  # pacman turn
            nextAgent = 1
            nextDepth = depth
        else:  # ghost turn
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            if nextAgent == 0:
                nextDepth = depth - 1
            else:
                nextDepth=depth

        legalActions = state.getLegalActions(agentIndex)
        successors = []
        for action in legalActions:
            successor = state.getNextState(agentIndex, action)
            successors.append(successor)
        scores = []
        for successor in successors:
            score = self.expectScore(successor, nextAgent, nextDepth)
            scores.append(score)
        if agentIndex == 0:
            return max(scores)
        else:
            return self.get_average_Score(scores)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    # ScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]

    #   food_evaluate
    food_evaluate = 0
    if len(currentFood.asList()) != 0:
        dist_with_food = []
        for food in list(currentFood.asList()):
            dist_with_food.append(util.manhattanDistance(food, currentPos))
        food_evaluate = -min(dist_with_food)

    #   ghost_evaluate
    ghost_evaluate = 0
    dist_with_ghost = []
    for ghost in currentGhostStates:
        dist_with_ghost.append(util.manhattanDistance(ghost.getPosition(), currentPos))
    min_dist_with_ghost = min(dist_with_ghost)
    if min_dist_with_ghost != 0:
        ghost_evaluate = 1 / min_dist_with_ghost

    return currentGameState.getScore() + food_evaluate + ghost_evaluate


# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState: GameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # NumofAgents = gameState.getNumAgents()

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # return childGameState.getScore()

        if childGameState.isWin():
            return float("inf")

        # 计算ghost_evaluate
        ghost_evaluate = 0
        dist_with_normal_ghost = []
        dist_with_scared_ghost = []
        min_dist_with_normal_ghost = 1000
        min_dist_with_scared_ghost = 0
        for newGhostState in newGhostStates:
            if newGhostState.scaredTimer > 4 and util.manhattanDistance(newGhostState.getPosition(), newPos) < 2:
                return float("inf")
            if newGhostState.scaredTimer == 0:
                if util.manhattanDistance(newGhostState.getPosition(), newPos) < 2:
                    return float("-inf")
                dist_with_normal_ghost.append(util.manhattanDistance(newGhostState.getPosition(), newPos))
            else:
                dist_with_scared_ghost.append(util.manhattanDistance(newGhostState.getPosition(), newPos))
        if dist_with_normal_ghost:
            min_dist_with_normal_ghost = min(dist_with_normal_ghost)
        if dist_with_scared_ghost:
            min_dist_with_scared_ghost = min(dist_with_scared_ghost)
        # ghost_evaluate = -100 * min_dist_with_normal_ghost + min_dist_with_scared_ghost
        ghost_evaluate = -1 * min_dist_with_normal_ghost

        # 计算food_evaluate
        dist_with_food = []
        for food in list(newFood.asList()):
            dist_with_food.append(util.manhattanDistance(food, newPos))

        food_evaluate = -min(dist_with_food)
        if (currentGameState.getNumFood() > childGameState.getNumFood()):
            food_evaluate += 100
        # if (currentGameState.getNumAgents() > childGameState.getNumAgents()):
        #     ghost_evaluate += 1000  # 如果吃掉了食物，增加100分
        # print("childGameState.getScore():", childGameState.getScore())
        # print("food_evaluate:", food_evaluate)
        # print("ghost_evaluate:", ghost_evaluate)

        return childGameState.getScore() + food_evaluate + ghost_evaluate
