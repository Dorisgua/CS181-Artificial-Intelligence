# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            state_count = util.Counter()
            for state in self.mdp.getStates():

                if self.mdp.isTerminal(state):
                    state_count[state] = 0
                    continue
                max_val = float('-inf')

                actions = self.mdp.getPossibleActions(state)

                for action in actions:
                    qvalue = self.computeQValueFromValues(state, action)
                    max_val = max(max_val, qvalue)
                    state_count[state] = max_val
            self.values = state_count

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        total_q_value = 0
        transition_model = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in transition_model:
            reward = self.mdp.getReward(state, action, nextState)
            Gamma = self.discount
            total_q_value = total_q_value + prob * (reward + Gamma * self.values[nextState])

        # 返回计算得到的Q值
        return total_q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # 如果当前状态是终止状态，返回 None   #which is the case at the
        #           terminal state, you should return None.
        if self.mdp.isTerminal(state):
            return None
        else:
            best_action = None
            best_val = float("-inf")

            for action in self.mdp.getPossibleActions(state):
                q_value = self.getQValue(state, action)

                if q_value > best_val:
                    best_action, best_val = action, q_value

            return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # 获取MDP中所有状态
        states = self.mdp.getStates()

        # 初始化剩余迭代次数
        iterations_left = self.iterations
        # 当剩余迭代次数大于 0 时，执行迭代更新
        while iterations_left > 0:
            # 对于每个状态
            for state in states:
                # 如果剩余迭代次数不足，提前结束迭代
                if iterations_left <= 0:
                    return

                # 如果当前状态不是终止状态
                if not self.mdp.isTerminal(state):
                    # 获取当前状态可执行的所有动作
                    actions = self.mdp.getPossibleActions(state)

                    # 获取当前状态所有可执行动作的最大Q值
                    max_q_value = max([self.getQValue(state, action) for action in actions])

                    # 更新当前状态的值为最大Q值
                    self.values[state] = max_q_value

                # 减少剩余迭代次数
                iterations_left -= 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {state: set() for state in self.mdp.getStates()}

        queue = util.PriorityQueue()
        # 查找每个状态的前继状态
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            for action in self.mdp.getPossibleActions(state):
                for nextState in self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[nextState[0]].add(state)

        temp_vals = {}

        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                q_values = [self.computeQValueFromValues(s, a) for a in actions]
                q_values.sort()
                max_q_value = q_values[-1]

                diff = abs(self.values[s] - max_q_value)
                queue.update(s, -diff)
                temp_vals[s] = max_q_value

        # 进行指定次数的迭代更新
        for _ in range(self.iterations):
            # 如果队列为空，跳出循环
            if queue.isEmpty():
                break
            s = queue.pop()
            # 更新 self.values 中状态 s 的值（如果它不是终止状态）。
            if not self.mdp.isTerminal(s):
                self.values[s] = temp_vals[s]

            # 对于该状态的前继状态
            for p in predecessors[s]:

                actions = self.mdp.getPossibleActions(p)
                q_values = [self.computeQValueFromValues(p, a) for a in actions]
                # q_values.sort()
                max_q_value = max(q_values)
                # 计算前继状态值与最大Q值之间的差异
                diff = abs(self.getValue(p) - max_q_value)
                temp_vals[p] = max_q_value

                # 如果差异超过阈值，则将状态加入优先队列中以进一步更新
                if diff > self.theta:
                    queue.update(p, -diff)
