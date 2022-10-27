import random
import numpy as np
import pandas as pd
import tensorflow as tf
import gym
from collections import deque
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Activation, Flatten, Conv1D, MaxPooling1D,Reshape
import matplotlib.pyplot as plt
import time

class DQN:
    ### TUNE CODE HERE ###
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=400000)
        self.gamma = 0.8
        self.epsilon = 0.2
        self.epsilon_min = 0.01
        self.epsilon_decay =  self.epsilon_min / 5000

        self.batch_size = 32
        self.train_start = 1000
        self.state_size = self.env.observation_space.shape[0]#*4
        self.action_size = self.env.action_space.n
        self.learning_rate = 0.0001

        self.evaluation_model = self.create_model()
        self.target_model = self.create_model()



    def create_model(self):
        model = Sequential()
        model.add(Dense(128*2, input_dim=self.state_size,activation='relu'))
        model.add(Dense(128*2, activation='relu'))
        model.add(Dense(128*2, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=self.learning_rate,decay=0.99,epsilon=1e-6))
        return model

    def choose_action(self, state, steps):
        if steps > 50000:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.evaluation_model.predict(state)[0])

    def remember(self, cur_state, action, reward, new_state, done):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = (cur_state, action, reward, new_state, done)
        self.memory.extend([transition])

        self.memory_counter += 1

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        mini_batch = random.sample(self.memory, self.batch_size)

        update_input = np.zeros((self.batch_size, self.state_size))
        update_target = np.zeros((self.batch_size, self.action_size))

        for i in range(self.batch_size):
            state, action, reward, new_state, done = mini_batch[i]
            target = self.evaluation_model.predict(state.reshape(1, self.state_size))[0]

            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.amax(self.target_model.predict(new_state.reshape(1, self.state_size))[0])

            update_input[i] = state
            update_target[i] = target

        self.evaluation_model.fit(update_input, update_target, batch_size=self.batch_size, epochs=1, verbose=0) # epochs 1

    def target_train(self):
        self.target_model.set_weights(self.evaluation_model.get_weights())
        return

    def visualize(self, reward, episode):
        plt.plot(episode, reward, 'ob-')
        plt.title('Average reward each 100 episode')
        plt.ylabel('Reward')
        plt.xlabel('Episodes')
        plt.grid()
        plt.show()

    #andrew add - not used
    def set_exploration_rate(self, episode):
        self.epsilon -= self.epsilon_decay
    ### END CODE HERE ###


def main():
    env = gym.make('Breakout-ram-v0')
    env = env.unwrapped
    #env.render()

    episodes = 500
    trial_len = 10000

    tmp_reward=0
    sum_rewards = 0

    graph_reward = []
    graph_episodes = []
    #added by andrew
    episode_reward = []
    episode_state = []

    steps = 0 #added by andrew

    dqn_agent = DQN(env=env)

    ####### Training ######
    ### START CODE HERE ###

    for episode in range(episodes):
        done = False
        frame = env.reset()
        sum_rewards = 0

        episode_reward = []
        episode_state = []

        while not done:
            #render
            #env.render()

            #select best action <-- from NN
            action = dqn_agent.choose_action(frame.reshape(1, dqn_agent.state_size), steps)

            #get reward and next state
            new_frame, reward, done, _ = env.step(action)

            #store episode data to train
            episode_reward.append(reward)
            episode_state.append((frame, action, new_frame, done))
            sum_rewards += reward

            #increment steps?
            steps += 1

            if (done):
                #episode has no next frame
                new_frame = np.zeros_like(frame)

            #store values
            dqn_agent.remember(frame, action, reward, new_frame, done)




        #batch = random.sample(dqn_agent.memory, dqn_agent.batch_size)
        #states_mb = np.array([each[0] for each in batch], ndmin=3)
        #actions_mb = np.array([each[1] for each in batch])
        #rewards_mb = np.array([each[2] for each in batch])
        #next_states_mb = np.array([each[3] for each in batch], ndmin=3)
        #dones_mb = np.array([each[4] for each in batch])

        #append data to lists
        graph_episodes.append(episode)
        graph_reward.append(sum_rewards)

        #discount the rewards
        episode_reward = discount_rewards(episode_reward, sum_rewards, dqn_agent.gamma)

        #update DQN
        update_input = np.zeros((len(episode_reward), dqn_agent.state_size))
        update_target = np.zeros((len(episode_reward), dqn_agent.action_size))
        for t in range(len(episode_reward)):
            frame, action, new_frame, done = episode_state[t]#batch[t]#episode_state[t]
            #create input state
            state = np.array(frame).reshape(1, dqn_agent.state_size)
            #create label
            #create next input state
            new_state = np.array(new_frame).reshape(1, dqn_agent.env.observation_space.shape[0])

            #assign reward
            target = np.zeros(dqn_agent.action_size)
            if (done):
                e_reward = episode_reward[t]
                target[action] = e_reward
            else:
                e_reward = episode_reward[t] + dqn_agent.gamma*np.max(dqn_agent.evaluation_model.predict(new_state)[0])
                target[action] = e_reward

            update_input[t] = state
            update_target[t] = target
        #update model
        dqn_agent.evaluation_model.fit(update_input, update_target, batch_size=dqn_agent.batch_size, epochs=1, verbose=0) # 1
        print("Hi Episode: %d Reward: %d" % (episode, sum_rewards))

        #experience replay
        for i in range(5):              # Only 1 replay
            dqn_agent.replay()

        #fixed q targets
        if (episode % 20 == 0):
            dqn_agent.target_train()

        if (episode % 250 == 0 and episode != 0):
            dqn_agent.visualize(graph_reward, graph_episodes)

        ### END CODE HERE ###

    dqn_agent.visualize(graph_reward, graph_episodes)

    #store values
    dqn_agent.evaluation_model.save_weights('DQN_Breakout_eval.h5')
    dqn_agent.target_model.save_weights('DQN_Breakout_target.h5')


def get_probability(graph_episodes, state, action):
    state_count = 0
    state_action_count = 0
    for state_action in graph_episodes:
        if (state_action[0].all(state)):
            state_count += 1
            if (state_action[1] == action):
                state_action_count += 1
    return (state_action_count/state_count)


def choose_action(dqn_agent, frame):
    exploration_rate_threshold = np.random.uniform(0,1)
    if (exploration_rate_threshold > dqn_agent.epsilon):
        a = dqn_agent.evaluation_model.predict(frame.reshape(1,dqn_agent.state_size))
        action = np.argmax(a)
    else:
        action = dqn_agent.env.action_space.sample()
    return action

def discount_rewards(rewards, sum_rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    sum_rewards = 0
    for t in reversed(range(0, len(rewards))):
        #if rewards[t] != 0: running_add = 0
        sum_rewards = sum_rewards * gamma + rewards[t]
        discounted_rewards[t] = sum_rewards
    return discounted_rewards
#def update_DQN():



if __name__ == '__main__':
    main()
