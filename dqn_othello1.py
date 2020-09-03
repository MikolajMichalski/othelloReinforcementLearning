# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import Softmax

EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size, env):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.90    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.9
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.env = env

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.add(Dense(self.action_size, activation="softmax"))
        model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate))
                      #optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            #return random.randrange(self.action_size)
            return random.choice(self.env.possible_actions)
        act_values = self.model.predict(state)
        possible_act_values = np.zeros((1, len(act_values[0])), float)
        for index in range(len(act_values[0])):
            if index in env.possible_actions:
                possible_act_values[0][index] = act_values[0][index]
        possible_action = np.argmax(possible_act_values[0])
        return(possible_action)
        #return np.argmax(act_values[0])  # returns action
    # def act(self, state):
    #     return random.choice(env.possible_actions)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    #env = gym.make('CartPole-v1')
    env = gym.make('Reversi8x8-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, env)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 1500
    total_reward = 0
    full_episodes_counter = 0
    iter_no = 0
    mean_reward = 0
    #for e in range(EPISODES):
    state = env.reset()
    while True:
        iter_no += 1
        #state = env.reset()
        state = np.reshape(state, [1, state_size])
        #for time in range(500):
            #env.render()
        action = agent.act(state)
        #if action in env.possible_actions:
        next_state, reward, done, _ = env.step(action)
        #else:
         #   continue
            #reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        if done:
            total_reward += reward
            full_episodes_counter += 1
            mean_reward = total_reward/full_episodes_counter
            print("episode: {}, mean_score: {:.4}, e: {:.2}"
                    .format(full_episodes_counter, mean_reward, agent.epsilon))
            if mean_reward > 40:
                print("Solved! Iterations played: {}, full episodes played: {}".format(iter_no, full_episodes_counter))
                agent.save("C:\\\\Users\\mikolajm\\PycharmProjects\\pythonProject\\save\\dqn-othello1-dqn.h5")
                break
            state = env.reset()
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")