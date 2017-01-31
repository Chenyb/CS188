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
        mdp = self.mdp
        values = self.values
        iterations = self.iterations
        discount = self.discount


        states = mdp.getStates() #get all states need to run
        copyOfValues = values.copy() #get a copy of value
        
        i = 0 #counter for while loop
        while i < iterations:
          i += 1
          for state in states:
            actions = mdp.getPossibleActions(state) # get actions of the current state
            v = []
            for action in actions:
              tempValue = 0
              for transitionState, prob in mdp.getTransitionStatesAndProbs(state, action):
                reward = mdp.getReward(state,action,transitionState)
                tempValue += prob *(reward + discount * copyOfValues[transitionState])
              v.append(tempValue)
            if v!= []:
              values[state] = max(v)
          copyOfValues = values.copy()
            


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
        mdp = self.mdp
        discount = self.discount
        values = self.values
        qValue = 0
        for transitionState, prob in mdp.getTransitionStatesAndProbs(state, action):
            reward = mdp.getReward(state, action, transitionState)
            qValue += prob * (reward + discount * values[transitionState])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        actions = mdp.getPossibleActions(state)
        resultAction = None
        score = -float("inf")

        for tempAction in actions:
            tempScore = self.computeQValueFromValues(state, tempAction)
            if score < tempScore:
                score = tempScore
                resultAction = tempAction
        return resultAction

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
        mdp = self.mdp
        values = self.values
        iterations = self.iterations
        discount = self.discount


        states = mdp.getStates() #get all states need to run
        copyOfValues = values.copy() #get a copy of value
        

        for i in range(iterations):
          state = states[i % len(states)]
          if mdp.isTerminal(state):
              pass
          else:
            actions = mdp.getPossibleActions(state) # get actions of the current state
            v = []
            for action in actions:
              tempValue = 0
              for transitionState, prob in mdp.getTransitionStatesAndProbs(state, action):
                reward = mdp.getReward(state,action,transitionState)
                tempValue += prob *(reward + discount * copyOfValues[transitionState])
              v.append(tempValue)
            if v!= []:
              values[state] = max(v)
          copyOfValues = values.copy()

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

        mdp = self.mdp
        values = self.values
        iterations = self.iterations
        discount = self.discount
        states = mdp.getStates() #get all states need to run
        copyOfValues = values.copy() #get a copy of value
        # Find predecessors of all states.
        predecessors = dict()
        for state in states:
          predecessors[state] = set()
        for state in states:
          actions = mdp.getPossibleActions(state)
          for action in actions:
            for transitionState, prob in mdp.getTransitionStatesAndProbs(state, action):
              predecessors[transitionState].add(state)
        # Initialize an empty priority queue
        priorityQueue = util.PriorityQueue()        
        # For each non-terminal state
        for state in states:
          if mdp.isTerminal(state) == False:
            v = []
            actions = mdp.getPossibleActions(state) # get actions of the current state
            for action in actions:
              tempValue = 0
              for transitionState, prob in mdp.getTransitionStatesAndProbs(state, action):
                reward = mdp.getReward(state,action,transitionState)
                tempValue += prob *(reward + discount * values[transitionState])
              v.append(tempValue)
            if v!= []:
              tempValue = max(v)
            diff = abs(values[state]-tempValue)
            priorityQueue.update(state, -diff)

        #For iteration......
        for _ in range(iterations):
          if priorityQueue.isEmpty() == False:
            state = priorityQueue.pop()
            actions = mdp.getPossibleActions(state) # get actions of the current state
            if mdp.isTerminal(state) == False:
              v = []  
              for action in actions:
                tempValue = 0
                for transitionState, prob in mdp.getTransitionStatesAndProbs(state, action):
                  reward = mdp.getReward(state,action,transitionState)
                  tempValue += prob *(reward + discount * values[transitionState])
                v.append(tempValue) 
              if v!= []:
                values[state] = max(v)   
            for predecessor in predecessors[state]:
              v = []
              actions = mdp.getPossibleActions(predecessor) # get actions of the current state
              for action in actions:
                tempValue = 0
                for transitionState, prob in mdp.getTransitionStatesAndProbs(predecessor, action):
                  reward = mdp.getReward(predecessor,action,transitionState)
                  tempValue += prob *(reward + discount * values[transitionState])
                v.append(tempValue)
              if v!= []:
                tempValue = max(v)
              diff = abs(values[predecessor]-tempValue)
              if diff > self.theta:
                priorityQueue.update(predecessor, -diff)




