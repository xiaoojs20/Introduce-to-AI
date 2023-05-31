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
        for i in range(self.iterations):
            nextValues = util.Counter() # A Counter is a dict with default 0
            for state in self.mdp.getStates():
                QValues = []
                if self.mdp.isTerminal(state):
                    QValues.append(0)
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    QValues.append(self.computeQValueFromValues(state, action))
                # self.values[state] = max(QValues)
                nextValues[state] = max(QValues)
            self.values = nextValues

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
        QValue = 0      
        statesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        for statesAndProb in statesAndProbs:
            nextState, prob = statesAndProb
            reward = self.mdp.getReward(state, action, nextState)
            QValue += prob*(reward + self.discount * self.values[nextState])
        return QValue
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        QValues = []
        for action in actions:
            QValues.append(self.computeQValueFromValues(state, action))

        maxQValueIndex = QValues.index(max(QValues))
        return actions[maxQValueIndex]
        util.raiseNotDefined()

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
        for i in range(self.iterations):
            states = self.mdp.getStates()
            QValues = []
            if self.mdp.isTerminal(states[i%len(states)]):
                QValues.append(0)
            actions = self.mdp.getPossibleActions(states[i%len(states)])
            for action in actions:
                QValues.append(self.computeQValueFromValues(states[i%len(states)], action))
            self.values[states[i%len(states)]] = max(QValues)

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
        # Compute predecessors of all states. Store in a dict.
        predecessors = {}
        for currentState in self.mdp.getStates():
            predecessor = set()
            for preState in self.mdp.getStates():
                actions = self.mdp.getPossibleActions(preState)
                for action in actions:
                    statesAndProbs = self.mdp.getTransitionStatesAndProbs(preState, action)
                    for statesAndProb in statesAndProbs:
                        nextState, prob = statesAndProb
                        if prob > 0 and nextState == currentState:
                            predecessor.add(preState)
            predecessors[currentState] = predecessor

        # Initialize an empty priority queue.
        pq = util.PriorityQueue()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                QValues = []
                for action in self.mdp.getPossibleActions(state):
                    QValues.append(self.computeQValueFromValues(state, action))
                diff = abs(self.values[state] - max(QValues))
                pq.push(state,-diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                break
            # update states that have a higher error
            updateState = pq.pop()
            if not self.mdp.isTerminal(updateState):
                QValues = []
                for action in self.mdp.getPossibleActions(updateState):
                    QValues.append(self.computeQValueFromValues(updateState, action))
                self.values[updateState] = max(QValues)
        
            for preState in predecessors[updateState]:
                QValues = []
                for action in self.mdp.getPossibleActions(preState):
                    QValues.append(self.computeQValueFromValues(preState, action))
                diff = abs(self.values[preState] - max(QValues))
                if diff > self.theta:
                    pq.update(preState,-diff)


