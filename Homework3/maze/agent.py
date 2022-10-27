import numpy as np
import pandas as pd
import random
import math


class Agent:
    ### START CODE HERE ###
    # ['u', 'd', 'l', 'r'] [0,1,2,3]
    def __init__(self, actions):
        self.actions = actions
        self.epsilon = 1    # noise
        self.discount = 0.9  # gamma
        self.learning_rate = 1 #alpha
        self.e_decay_rate = 0.005
        self.a_decay_rate = 0.005
        self.k = 2    # exploration function
        self.Q = np.zeros((72, len(self.actions)))
        self.R = np.zeros((72, len(self.actions)))
        self.S_ = np.zeros((72, len(self.actions)))
        self.previous_sa_pairs = set([])
        self.previous_seen_states = {}
        self.last_saw_state = {}

    def getIndex(self, observation):
        #print("IN GET INDEX" + str(observation))
        treasureFound = None
        if observation[4] == False:
            treasureFound = 0
        else:
            treasureFound = 36

        x = math.floor(observation[0]/40)
        y = math.floor(observation[1]/40)

        return ((y * 6) + x) + treasureFound

    def choose_action(self, observation, episode):
        #print("IN CHOOSE ACTION" + str(observation))
        #print("observation: " + str(observation) + " index is : " + str(self.getIndex(observation)))
        action = None
        s = self.getIndex(observation)
        random_ep = np.random.uniform(0,1)
        '''
        if  self.epsilon > random_ep:
            action = np.random.choice(self.actions)
            listActions = []
            for a in range(4):
                #print("IN CHOOSE ACTION")
                if (s,a) not in self.previous_seen_states:
                    listActions.append(a)
            if len(listActions) is not 0:
                    action = np.random.choice(listActions)
        '''
        if True:
        #    print("IN NOT RANDOM")
            #print(self.Q[s])
            max = -999999
            listActions = []
            for a in range(4):
                    if (s,a) not in self.previous_seen_states:
                    #    print("Q Value: " +  str(((1 - self.epsilon) * self.Q[s][a]) + ((self.k))) + " = " + str(self.Q[s][a]) + " + " + str(self.k) + " * " + str(self.epsilon))
                        q_value = ((1 - self.epsilon) * self.Q[s][a]) + ((self.k))

                    else:
                        q_value = ((1 - self.epsilon) * self.Q[s][a]) + self.epsilon * (self.k/6)/self.previous_seen_states[(s,a)]
                    #    print("Q Value: " +  str(q_value) + " = " + str(self.Q[s][a]) + " + " + str((self.k/6)/self.previous_seen_states[(s,a)]) + " * " + str(self.epsilon))
                    if q_value > max:
                        max = q_value
            #print("Max is value: " +  str(max))
            for a in range(4):
                    if (s,a) not in self.previous_seen_states:
                        #print("Q Value: " +  str(self.Q[s][a] + ((self.k) * self.epsilon)) + " = " + str(self.Q[s][a]) + " + " + str(self.k) + " * " + str(self.epsilon))
                        q_value = ((1 - self.epsilon) * self.Q[s][a]) + ((self.k))

                    else:
                        q_value = ((1 - self.epsilon) * self.Q[s][a]) + self.epsilon * (self.k/6)/self.previous_seen_states[(s,a)]
                        #print("Q Value: " +  str(q_value) + " = " + str(self.Q[s][a]) + " + " + str((self.k/2)/self.previous_seen_states[(s,a)]) + " * " + str(self.epsilon))
                    if max == q_value:
                        listActions.append(a)

            #print("Max is actions: " +  str(listActions))
            action = np.random.choice(listActions)
            #print("Action chosen is: " + str(action))
            #print(self.epsilon)
        return action

    def reduce_noise(self,episode):
        self.epsilon = 0.001 + 0.99 * (np.exp(-self.e_decay_rate*episode))
        self.learning_rate = 0.001 + 0.99 * (np.exp(-self.a_decay_rate*episode))

    def UpdateQ(self, s, a, s_, r):
        #print("IN UPDATEQ")
        stateEntry = self.getIndex(s)
        indexNextState = self.getIndex(s_)
        s_qValues = self.Q[indexNextState]
        # for state "stateEntry" and action "a"
        #print("Old value is: " + str(self.Q[stateEntry][a]))
        #print(str(self.Q[stateEntry][a]) + " + " + str(self.learning_rate) + " * (" + str(r) + " + " + str(self.discount) + " * " + str(self.getMaxQ(indexNextState)) + " - " + str(self.Q[stateEntry][a]) + ")")
        self.Q[stateEntry][a] = self.Q[stateEntry][a] + self.learning_rate * ((r + self.discount*(self.getMaxQ(indexNextState)) - self.Q[stateEntry][a]))
        #print("New value is: " + str(self.Q[stateEntry][a]))

    def HallucinateQ(self, s, a, s_, r):
        # for state "stateEntry" and action "a"
        self.Q[s][a] = self.Q[s][a] + self.learning_rate * ((r + self.discount*(self.getMaxQ(s_)) - self.Q[s][a]))


    def UpdateM(self, s, a, s_, r, episode):
        stateEntry = self.getIndex(s)
        nextStateEntry = self.getIndex(s_)
        if (stateEntry,a) not in self.previous_seen_states:
            self.previous_seen_states[(stateEntry,a)] = 1
            self.last_saw_state[(stateEntry,a)] = episode
        else:
            self.previous_seen_states[(stateEntry,a)] += 1
            self.last_saw_state[(stateEntry,a)] = episode

        # for state "stateEntry" and action "a"
        self.R[stateEntry][a] = r
        self.S_[stateEntry][a] = nextStateEntry

    def getMaxQ(self, s):
        max = -999999
        for a in range(4):
            if (s,a) not in self.previous_seen_states:
                #print("Q Value: " +  str(self.Q[s][a] + self.k) + " = " + str(self.Q[s][a]) + " + " + str(self.k))
                q_value = self.Q[s][a]

            else:
                #print("Q Value: " +  str(self.Q[s][a] + ((self.k/2)/self.previous_seen_states[(s,a)])) + " = " + str(self.Q[s][a]) + " + " + str(self.k/2) + "/" + str(self.previous_seen_states[(s,a)]))
                q_value = self.Q[s][a]
            if q_value > max:
                max = q_value
        return max

    def getModel(self, s, a):
        # for state "stateEntry" and action "a"
        return self.S_[s][a], self.R[s][a]

    def Hallucinate(self):
        for n in range(2):
            sample = random.choice(list(self.previous_seen_states.keys()))
            s = sample[0]
            a = sample[1]
            s_, r = self.getModel(s, a)
            s_ = int(s_)
            #print("HALL: " + str(s) + " " + str(a) + " " + str(s_) + " " + str(r))
            self.HallucinateQ(s, a, s_, r)
        return


    ### END CODE HERE ###
