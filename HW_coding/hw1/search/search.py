# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from collections import deque
    if problem.isGoalState(problem.getStartState()):
        return []
    ready_to_be_in = util.Stack()
    start_state = problem.getStartState()
    visited = []
    ready_to_be_in.push((start_state, list()))

    while 1:
        if ready_to_be_in.isEmpty():
            return []
        else:
            # print("pop")
            current_state = ready_to_be_in.pop()
            current_position, actions = current_state[0], current_state[1]
            # print(current_state)
            # print(actions)
            if problem.isGoalState(current_position):#如果是目标的话
                return list(actions)
            if current_position in visited:
                continue
            else:#current_position not in visited:
                visited.append(current_position)
                children = problem.getSuccessors(current_position)
                for child_position, action, _ in children:
                    if child_position not in visited:
                        # print("hi")
                        new_actions = actions.copy()
                        new_actions.append(action)
                        ready_to_be_in.push((child_position, new_actions))
                        # print(list(new_actions))


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()):
        return []
    ready_to_be_in = util.Queue()
    start_position = problem.getStartState()
    visited = []
    ready_to_be_in.push((start_position, list()))

    while 1:
        if ready_to_be_in.isEmpty():
            return []
        else:
            # print("pop")
            current_state = ready_to_be_in.pop()
            current_position, actions = current_state[0], current_state[1]
            # print(current_state)
            # print(actions)
            if problem.isGoalState(current_position):#如果是目标的话
                return list(actions)
            if current_position in visited:
                continue
            else:#current_position not in visited:
                visited.append(current_position)
                children = problem.getSuccessors(current_position)
                for child_position, action, _ in children:
                    if child_position not in visited:
                        # print("hi")
                        new_actions = actions.copy()
                        new_actions.append(action)
                        ready_to_be_in.push((child_position, new_actions))
                        # print(list(new_actions))

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    if problem.isGoalState(problem.getStartState()):
        return []
    ready_to_be_in = util.PriorityQueue()
    start_position = problem.getStartState()
    visited = []
    ready_to_be_in.push((start_position, list(), 0), 0)

    while 1:
        if ready_to_be_in.isEmpty():
            return []
        else:
            # print("pop")
            current_state = ready_to_be_in.pop()
            current_position, actions, total_cost = current_state[0], current_state[1], current_state[2]
            # print(current_state)
            # print(actions)
            if problem.isGoalState(current_position):  # 如果是目标的话
                return list(actions)
            if current_position in visited:
                continue
            else:  # current_position not in visited:
                visited.append(current_position)
                children = problem.getSuccessors(current_position)
                for child_position, action, cost in children:
                    if child_position not in visited:
                        # print("hi")
                        new_actions = actions.copy()
                        new_actions.append(action)
                        new_total_cost = total_cost + cost
                        ready_to_be_in.push((child_position, new_actions, new_total_cost), new_total_cost)
                        # print(list(new_actions))
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    if problem.isGoalState(problem.getStartState()):
        return []
    ready_to_be_in = util.PriorityQueue()
    start_position = problem.getStartState()
    visited = []
    ready_to_be_in.push((start_position, list(), 0), 0)
    while 1:
        if ready_to_be_in.isEmpty():
            return []
        else:
            # print("pop")
            current_state = ready_to_be_in.pop()
            current_position, actions, total_cost = current_state[0], current_state[1], current_state[2]
            # print(current_state)
            # print(actions)
            if problem.isGoalState(current_position):  # 如果是目标的话
                return list(actions)
            if current_position in visited:
                continue
            else:  # current_position not in visited:
                visited.append(current_position)
                children = problem.getSuccessors(current_position)
                for child_position, action, cost in children:
                    if child_position not in visited:
                        # print("hi")
                        new_actions = actions.copy()
                        new_actions.append(action)
                        new_total_cost = total_cost + cost
                        new_total_cost_and_h = new_total_cost + heuristic(child_position, problem)
                        ready_to_be_in.push((child_position, new_actions, new_total_cost), new_total_cost_and_h)
                        # print(list(new_actions))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
