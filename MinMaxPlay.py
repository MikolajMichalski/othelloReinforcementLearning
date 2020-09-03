import gym
import random
import numpy as np
env = gym.make('Reversi8x8-v0')
env.reset()


EPISODES = 500
white_wins = 0

for i_episode in range(EPISODES):

    observation = env.reset()
    for t in range(100):
        enables = env.possible_actions
        # if nothing to do ,select pass
        if len(enables)==0:
            action = env.board_size**2 + 1
        # random select (update learning method here)
        else:
            action = random.choice(enables)
        observation, reward, done, info = env.step(action)

        #env.render()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            black_score = len(np.where(env.state[0,:,:]==1)[0])
            white_score = len(np.where(env.state[1,:,:]==1)[0])
            print("Black score: {} \nWhite score: {}".format(black_score, white_score))
            if black_score > white_score:
                print("BLACK PLAYER WINS")
                print("White player wins: {}/{} % {:.4}".format(white_wins, i_episode+1, white_wins/(i_episode+1)*100))
            elif black_score<=white_score:
                white_wins += 1
                print("WHITE PLAYER WINS!")
                print("White player wins: {}/{} % {:.4}".format(white_wins, i_episode+1, (white_wins/(i_episode+1)*100)))
            break




