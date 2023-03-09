import random
import pickle
import numpy as np


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
        self.history = []

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        with open(filename + '.pickle', 'rb') as handle:
            self.q = pickle.load(handle)

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # Store data
        with open(filename + '.pickle', 'wb') as handle:
            pickle.dump(self.q, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # if return_q is set to True return (action, q) instead of just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 

        if random.uniform(0, 1) < self.epsilon:
            myList = list(self.actions)
            # myList.append(1)
            random.shuffle(myList)
            chosen_action = myList[0]
            print("hello")
        else:
            optimal = []
            max = -np.inf
            for action in self.actions:
                value = self.getQ(state, action)
                if value > max: 
                    optimal.clear()
                    optimal.append( action )
                    max = value
            
            random.shuffle(optimal)
            chosen_action = optimal[0]
            print("OPTIMAL")

        if return_q == True:
            value = self.getQ(state, chosen_action)
            return (chosen_action, value)
        print(chosen_action)
        self.history.append( chosen_action )
        print(np.mean(self.history))
        return chosen_action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        # Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        max = 0
        for action in self.actions:
            value = self.getQ(state2, action)
            if value > max: 
                max = value

        if self.q.get((state1,action1)) == None:
            self.q[(state1,action1)] = reward
        
        self.q[(state1,action1)] += self.alpha * (reward  + self.gamma * max - self.q[(state1,action1)])
