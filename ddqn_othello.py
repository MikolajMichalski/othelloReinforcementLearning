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
import sys

EPISODES = 1000
outputFilePath = "C:\\\\Users\\mikolajm\\OneDrive - Intel Corporation\\Desktop\\MGR\\pythonProject\\save\\output-new1.txt"

def writeStdOutputToFile(filePath, text):
    original_std_out = sys.stdout
    with open(filePath, "a") as f:
        sys.stdout = f
        print(text)
        sys.stdout = original_std_out


class DDQNAgent:
    def __init__(self, state_size, action_size, env):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.learning_rate = 0.005
        self.model = self.initiate_model()
        self.target_model = self.initiate_model()
        self.env = env
        self.sync_target_model()

    def initiate_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.add(Dense(self.action_size, activation="softmax"))
        model.compile(loss='mse', optimizer=SGD(lr=self.learning_rate))
        return model

    def replay_buffer_save(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action_to_make(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.env.possible_actions)
        act_values = self.model.predict(state)
        possible_act_values = np.zeros((1, len(act_values[0])), float)
        for index in range(len(act_values[0])):
            if index in env.possible_actions:
                possible_act_values[0][index] = act_values[0][index]
        possible_action = np.argmax(possible_act_values[0])
        return possible_action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_weights(self, name):
        self.model.load_weights(name)

    def save_weights(self, name):
        self.model.save_weights(name)

    def sync_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())


if __name__ == "__main__":
    env = gym.make('Reversi8x8-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    n_episodes_deque_size = 10
    last_n_episodes_scores = deque(maxlen=n_episodes_deque_size)
    agent = DDQNAgent(state_size, action_size, env)
    done = False
    batch_size = 30

    total_reward = 0
    full_episodes_counter = 0
    iter_no = 0
    mean_reward = 0.0
    max_mean_reward = 0.0
    random_episode_no = 0
    load_weights = True
    random_episodes_mean_reward = 0.0
    random_episodes_total_reward = 0.0
    random_iterations_to_play = 900
    games_won = 0
    max_reward = 0

    state = env.reset()     ### RESET THE ENVIRONMENT ON START OF THE GAME
    if load_weights:  ### LOAD WEIGHTS IF PLAYING ON TRAINED MODEL
        agent.epsilon = 0.0 ### EPSILON = 0 TO ALWAYS USE ACTIONS BASED ON MODEL CHOOSE
        agent.load_weights("save/othello-dqn-target-new-minmax-2.h5")
        while True:

            iter_no += 1
            state = np.reshape(state, [1, state_size])
            action = agent.get_action_to_make(state)
            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            if done:
                black_score = sum(1 for i in next_state[0] if i == 1)
                white_score = sum(1 for i in next_state[0] if i == -1)
                state = env.reset() ### RESET THE STATE IF EPISODE FINISHED
                full_episodes_counter += 1
                if reward > 0:
                    games_won +=1

                    print("Games won/total: {}/{}, win %: {:.4}%, last score: {}, black/white: {}/{}".format(games_won, full_episodes_counter, games_won/full_episodes_counter*100, reward, black_score, white_score))
                else:
                    print("Games won/total: {}/{}, win %: {:.4}%, last score: {}, black/white: {}/{}".format(games_won, full_episodes_counter,
                                                                         games_won / full_episodes_counter * 100, reward, black_score, white_score))
                if full_episodes_counter == EPISODES:
                    break

    else:  ### IF WEIGHTS ARE NOT LOADED - TRAIN THE MODEL
        while True:
            ## RANDOM PLAY UNTIL NUMBER OF ELEMENTS IN MEMORY IS EQUAL MINIBATCH SIZE
            iter_no += 1
            state = np.reshape(state, [1, state_size])

            action = agent.get_action_to_make(state) ### CHOOSE ACTION BASED ON EPSILON GREEDY POLICY

            next_state, reward, done, _ = env.step(action) ### PERFORM CHOOSEN ACTION
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.replay_buffer_save(state, action, reward, next_state, done)  ### SAVE STATE, ACTION, REWARD, NEXT_STATE AND DONE IN REPLY BUFFER
            state = next_state #### SET STATE TO NEXT STATE
            random_episodes_total_reward = float(reward)
            if done:   ### RESET STATE IF EPISODE FINISHED
                env.render()
                state = env.reset()
            if len(agent.memory) > random_iterations_to_play: #batch_size:  ### PLAY SPECIFIED NUMER OF EPISODES WITH RANDOM POLICY.
                agent.replay(batch_size)
                if done:  ### IF DONE ADD THE FINAL REWARD OF THE EPISODE TO TOTAL_REWARD AND INCREMENT FULL_EPISODES_COUNTER

                    full_episodes_counter += 1
                    last_n_episodes_scores.append(reward)
                    if full_episodes_counter >= 10:
                        mean_reward = sum(1 for i in last_n_episodes_scores if i == 1)/len(last_n_episodes_scores)
                        if mean_reward == 1.0:
                            n_episodes_deque_size += 1

                    if reward > 0:
                        agent.sync_target_model()
                        agent.save_weights("C:\\\\Users\\mikolajm\\OneDrive - Intel Corporation\\Desktop\\MGR\\pythonProject\\save\\othello-dqn-new-minmax-2.h5")
                    print("Trained episode: {}, current score: {} mean_score: {:.4}, e: {:.2}, iterations (steps): {}".format(full_episodes_counter, reward, mean_reward, agent.epsilon, iter_no)) ### PRINT FINAL INFO ABOUT EPISODES
                    writeStdOutputToFile(outputFilePath,"Trained episode: {}, current score: {} mean_score: {:.4}, e: {:.2}, iterations (steps): {}".format(full_episodes_counter, reward, mean_reward, agent.epsilon, iter_no))  ### PRINT FINAL INFO ABOUT EPISODES
                    if max_mean_reward != 1.0:
                        if mean_reward > max_mean_reward:  ### IF CURRENT MEAN_REWARD IS THE HIGHEST THEN SAVE THE WEIGHTS
                            max_mean_reward = mean_reward
                            print("New max mean reward!")
                            writeStdOutputToFile(outputFilePath, "New max mean reward")
                            agent.target_model.save(
                                "C:\\\\Users\\mikolajm\\OneDrive - Intel Corporation\\Desktop\\MGR\\pythonProject\\save\\othello-dqn-target-new-minmax-2.h5")
                    else:
                        if mean_reward == 1.0:
                            print("New max mean reward!")
                            writeStdOutputToFile(outputFilePath, "New max mean reward!")
                            agent.target_model.save(
                                "C:\\\\Users\\mikolajm\\OneDrive - Intel Corporation\\Desktop\\MGR\\pythonProject\\save\\othello-dqn-target-new-minmax-2.h5")

                if mean_reward == 1 and n_episodes_deque_size == 20:
                    print("Solved! Iterations played: {}, full episodes played: {}".format(iter_no, full_episodes_counter))
                    writeStdOutputToFile(outputFilePath, "Solved! Iterations played: {}, full episodes played: {}".format(iter_no, full_episodes_counter))
                    agent.target_model.save("C:\\\\Users\\mikolajm\\OneDrive - Intel Corporation\\Desktop\\MGR\\pythonProject\\save\\othello-dqn-target-new-minmax-2.h5")
                    break
            elif done:
                random_episode_no += 1

                print("Random episodes played: {}, random episode score: {:.4} iterations (steps): {}.".format(
                    random_episode_no, random_episodes_total_reward, iter_no))
                writeStdOutputToFile(outputFilePath,
                                     "Random episodes played: {}, random episode score: {:.4} iterations (steps): {}.".format(
                                         random_episode_no, random_episodes_total_reward, iter_no))
                random_episodes_total_reward = 0.
