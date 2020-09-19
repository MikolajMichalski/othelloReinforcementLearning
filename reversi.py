"""
Game of Reversi
"""

from six import StringIO
import sys
import gym
from gym import spaces
import numpy as np
from gym import error
from gym.utils import seeding
from copy import deepcopy
from collections import deque
from termcolor import colored
import collections

def make_random_policy(np_random):
    def random_policy(state, player_color):
        possible_places = ReversiEnv.get_possible_actions(state, player_color)
        # No places left
        if len(possible_places) == 0:
            return d**2 + 1
        a = np_random.randint(len(possible_places))
        return possible_places[a]
    return random_policy

class ReversiEnv(gym.Env):
    """
    Reversi environment. Play against a fixed opponent.
    """
    BLACK = 0
    WHITE = 1
    metadata = {"render.modes": ["ansi","human"]}
    MINMAX_DEPTH = 2
    OPPONENT_POLICY_TYPE = 'minmax'

    def __init__(self, player_color, opponent, observation_type, illegal_place_mode, board_size):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: An opponent policy
            observation_type: State encoding
            illegal_place_mode: What to do when the agent makes an illegal place. Choices: 'raise' or 'lose'
            board_size: size of the Reversi board
        """
        assert isinstance(board_size, int) and board_size >= 1, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size

        colormap = {
            'black': ReversiEnv.BLACK,
            'white': ReversiEnv.WHITE,
        }
        try:
            self.player_color = colormap[player_color]
        except KeyError:
            raise error.Error("player_color must be 'black' or 'white', not {}".format(player_color))

        self.opponent = opponent

        assert observation_type in ['numpy3c']
        self.observation_type = observation_type

        assert illegal_place_mode in ['lose', 'raise']
        self.illegal_place_mode = illegal_place_mode

        if self.observation_type != 'numpy3c':
            raise error.Error('Unsupported observation type: {}'.format(self.observation_type))

        # One action for each board position and resign and pass
        self.action_space = spaces.Discrete(self.board_size ** 2 + 2)
        observation = self.reset()
        self.observation_space = spaces.Box(np.zeros(observation.shape), np.ones(observation.shape))

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Update the random policy if needed
        if isinstance(self.opponent, str):
            if self.opponent == 'random':
                self.opponent_policy = make_random_policy(self.np_random)
            else:
                raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))
        else:
            self.opponent_policy = self.opponent

        return [seed]

    def reset(self):
        # init board setting
        self.state = np.zeros((3, self.board_size, self.board_size))
        centerL = int(self.board_size/2-1)
        centerR = int(self.board_size/2)
        self.state[2, :, :] = 1.0
        self.state[2, (centerL):(centerR+1), (centerL):(centerR+1)] = 0
        self.state[0, centerR, centerL] = 1
        self.state[0, centerL, centerR] = 1
        self.state[1, centerL, centerL] = 1
        self.state[1, centerR, centerR] = 1
        self.to_play = ReversiEnv.BLACK
        self.possible_actions = ReversiEnv.get_possible_actions(self.state, self.to_play)
        self.done = False

        # Let the opponent play if it's not the agent's turn
        if self.player_color != self.to_play:
            a = self.opponent_policy(self.state)
            ReversiEnv.make_place(self.state, a, ReversiEnv.BLACK)
            self.to_play = ReversiEnv.WHITE

        obs = self.getCurrentObservations(self.state)
        return obs

    def step(self, action):
        reward = 0
        passPlace_counter = 0;
        assert self.to_play == self.player_color
        # If already terminal, then don't do anything
        if self.done:
            obs = self.getCurrentObservations(self.state)
            return obs, 0., True, {'state': self.state}
        if ReversiEnv.pass_place(self.board_size, action):
            passPlace_counter += 1
            pass
        elif ReversiEnv.resign_place(self.board_size, action):
            obs = self.getCurrentObservations(self.state)
            return obs, -10., True, {'state': self.state}
        elif not ReversiEnv.valid_place(self.state, action, self.player_color):
            if self.illegal_place_mode == 'raise':
                raise
            elif self.illegal_place_mode == 'lose':
                # Automatic loss on illegal place
                self.done = True
                obs = self.getCurrentObservations(self.state)
                #return self.state, -1., True, {'state': self.state}
                return obs, -10., True, {'state': self.state}
            else:
                raise error.Error('Unsupported illegal place action: {}'.format(self.illegal_place_mode))
        else:
            ReversiEnv.make_place(self.state, action, self.player_color)

        #self.render()
        # Opponent play

        tmporary_state = deepcopy(self.state)
        if self.OPPONENT_POLICY_TYPE == "minmax":
            a = self.best_action(tmporary_state)
        else:
            a = self.opponent_policy(self.state, 1 - self.player_color) ####### USE MINIMAX TO CHOOSE ACTION

        # Making place if there are places left
        if a is not None:
            if ReversiEnv.pass_place(self.board_size, a):
                passPlace_counter += 1

                if passPlace_counter <= 1:
                    pass
                else:
                    self.done = True
                    obs = self.getCurrentObservations(self.state)
                    score = 0.
                    score = self.GetCurrentScore(self.state, reward)
                    if sum(1 for i in obs if i==1) > sum(1 for i in obs if i==-1): #score >= 32:
                        reward = 1.
                    else:
                        reward = -1

                    return obs, reward, self.done, {'state': self.state}

            elif ReversiEnv.resign_place(self.board_size, a):
                obs = self.getCurrentObservations(self.state)
                return obs, 0, True, {'state': self.state}
            elif not ReversiEnv.valid_place(self.state, a, 1 - self.player_color):
                if self.illegal_place_mode == 'raise':
                    raise
                elif self.illegal_place_mode == 'lose':
                    # Automatic loss on illegal place
                    self.done = True
                    obs = self.getCurrentObservations(self.state)
                    return obs, 0, True, {'state': self.state}
                else:
                    raise error.Error('Unsupported illegal place action: {}'.format(self.illegal_place_mode))
            else:
                ReversiEnv.make_place(self.state, a, 1 - self.player_color)

        reward = -1.0

        self.possible_actions = ReversiEnv.get_possible_actions(self.state, self.player_color)

        obs = self.getCurrentObservations(self.state)

        self.done = self.isFinished(self.state)
        if self.done:
            score = 0.
            score = self.GetCurrentScore(self.state, reward)
            if score >= 32:
                reward = 1.
            elif score < 32:
                reward = -1
        #self.render()
        return obs, reward, self.done, {'state': self.state}


    def GetCurrentScore(self, state, current_reward):
        #reward = 0
        for x in range(self.state.shape[2]):
            for y in range(self.state.shape[2]):
                if self.state[0, x, y] == 1:
                    current_reward += 1
        return current_reward

    def render(self, mode='human',  close=False):
        if close:
            return

        board = self.state
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        outfile.write(colored(' ' * 7))
        for j in range(board.shape[1]):
            outfile.write(colored(' ' +  str(j + 1) + '  | ', 'grey'))
        outfile.write('\n')
        outfile.write(' ' * 5)
        outfile.write(colored('-' * (board.shape[1] * 6 - 1), "grey"))
        outfile.write('\n')
        for i in range(board.shape[1]):
            outfile.write(colored(' ' +  str(i + 1) + '  |', "grey"))
            for j in range(board.shape[1]):
                if board[2, i, j] == 1:
                    outfile.write(colored('  O  ', "blue"))
                elif board[0, i, j] == 1:
                    outfile.write(colored('  B  ', "green"))
                else:
                    outfile.write(colored('  W  ', "red"))
                outfile.write(colored('|', "grey"))
            outfile.write('\n')
            outfile.write(' ' )
            outfile.write(colored('-' * (board.shape[1] * 7 - 1), "grey"))
            outfile.write('\n')

        if mode != 'human':
            return outfile

    @staticmethod
    def resign_place(board_size, action):
        return action == board_size ** 2

    @staticmethod
    def pass_place(board_size, action):
        return action == board_size ** 2 + 1

    @staticmethod
    def get_possible_actions(board, player_color):
        actions=[]
        d = board.shape[-1]
        opponent_color = 1 - player_color
        for pos_x in range(d):
            for pos_y in range(d):
                if (board[2, pos_x, pos_y]==0):
                    continue
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if(dx == 0 and dy == 0):
                            continue
                        nx = pos_x + dx
                        ny = pos_y + dy
                        n = 0
                        if (nx not in range(d) or ny not in range(d)):
                            continue
                        while(board[opponent_color, nx, ny] == 1):
                            tmp_nx = nx + dx
                            tmp_ny = ny + dy
                            if (tmp_nx not in range(d) or tmp_ny not in range(d)):
                                break
                            n += 1
                            nx += dx
                            ny += dy
                        if(n > 0 and board[player_color, nx, ny] == 1):
                            actions.append(pos_x * d + pos_y)
        if len(actions)==0:
            actions = [d**2 + 1]
        return actions

    @staticmethod
    def valid_reverse_opponent(board, coords, player_color):
        '''
        check whether there is any reversible places
        '''
        d = board.shape[-1]
        opponent_color = 1 - player_color
        pos_x = coords[0]
        pos_y = coords[1]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if(dx == 0 and dy == 0):
                    continue
                nx = pos_x + dx
                ny = pos_y + dy
                n = 0
                if (nx not in range(d) or ny not in range(d)):
                    continue
                while(board[opponent_color, nx, ny] == 1):
                    tmp_nx = nx + dx
                    tmp_ny = ny + dy
                    if (tmp_nx not in range(d) or tmp_ny not in range(d)):
                        break
                    n += 1
                    nx += dx
                    ny += dy
                if(n > 0 and board[player_color, nx, ny] == 1):
                    return True
        return False

    @staticmethod
    def valid_place(board, action, player_color):
        coords = ReversiEnv.action_to_coordinate(board, action)
        # check whether there is any empty places
        if board[2, coords[0], coords[1]] == 1:
            # check whether there is any reversible places
            if ReversiEnv.valid_reverse_opponent(board, coords, player_color):
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def make_place(board, action, player_color):
        coords = ReversiEnv.action_to_coordinate(board, action)

        d = board.shape[-1]
        opponent_color = 1 - player_color
        pos_x = coords[0]
        pos_y = coords[1]

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if(dx == 0 and dy == 0):
                    continue
                nx = pos_x + dx
                ny = pos_y + dy
                n = 0
                if (nx not in range(d) or ny not in range(d)):
                    continue
                while(board[opponent_color, nx, ny] == 1):
                    tmp_nx = nx + dx
                    tmp_ny = ny + dy
                    if (tmp_nx not in range(d) or tmp_ny not in range(d)):
                        break
                    n += 1
                    nx += dx
                    ny += dy
                if(n > 0 and board[player_color, nx, ny] == 1):
                    nx = pos_x + dx
                    ny = pos_y + dy
                    while(board[opponent_color, nx, ny] == 1):
                        board[2, nx, ny] = 0
                        board[player_color, nx, ny] = 1
                        board[opponent_color, nx, ny] = 0
                        nx += dx
                        ny += dy
                    board[2, pos_x, pos_y] = 0
                    board[player_color, pos_x, pos_y] = 1
                    board[opponent_color, pos_x, pos_y] = 0
        return board

    @staticmethod
    def coordinate_to_action(board, coords):
        return coords[0] * board.shape[-1] + coords[1]

    @staticmethod
    def action_to_coordinate(board, action):
        return action // board.shape[-1], action % board.shape[-1]

    # @staticmethod
    # def game_finished(board):
    #     # Returns 1 if player 1 wins, -1 if player 2 wins and 0 otherwise
    #     d = board.shape[-1]
    #
    #     player_score_x, player_score_y = np.where(board[0, :, :] == 1)
    #     player_score = len(player_score_x)
    #     opponent_score_x, opponent_score_y = np.where(board[1, :, :] == 1)
    #     opponent_score = len(opponent_score_x)
    #     if player_score == 0:
    #         return -5
    #     elif opponent_score == 0:
    #         return 5
    #     else:
    #         free_x, free_y = np.where(board[2, :, :] == 1)
    #         if free_x.size == 0:
    #             if player_score > (d**2)/2:
    #                 return 5
    #             elif player_score == (d**2)/2:
    #                 return 5
    #             else:
    #                 return -5
    #         else:
    #             return 0
    #     return 0

    def isFinished(self, board):
        for x in range(board.shape[-1]):
            for y in range(board.shape[-1]):
                if board[2, x, y] == 1:
                    return False
        return True

    def getCurrentObservations(self, state):
        obs = np.empty([state.shape[-1], state.shape[-1]])
        for x in range(state.shape[-1]):
            for y in range(state.shape[-1]):
                if state[2, x, y] == 1:
                    obs[x, y] = 0
                else:
                    if state[0, x, y] == 1:
                        obs[x, y] = 1
                    else:
                        if state[1, x, y] == 1:
                            obs[x, y] = -1
        obs = np.concatenate(obs)
        return obs

    def calculate_min_max_action(self, game_state, depth, maximizing_player):
        tmp_state = deepcopy(game_state)

        if maximizing_player:
            current_player_color = 1
        else:
            current_player_color = 0
        possible_actions_tmp = self.get_possible_actions(tmp_state, current_player_color)

        if depth == self.MINMAX_DEPTH or (len(possible_actions_tmp)==1 and any(possible_actions_tmp) > 64):
            tmp_obs = self.getCurrentObservations(tmp_state)
            score = sum(1 for i in self.getCurrentObservations(tmp_state) if i==-1)
            return score

        if maximizing_player:
            best_score = -9999
            possible_actions_tmp = self.get_possible_actions(tmp_state, current_player_color)
            for action in possible_actions_tmp:
                tmp_state1 = deepcopy(tmp_state)
                if action != 65:
                    tmp_state = ReversiEnv.make_place(tmp_state, action, 1)
                score = self.calculate_min_max_action(tmp_state, depth + 1, False)
                tmp_state = tmp_state1
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = 9999
            possible_actions_tmp = self.get_possible_actions(tmp_state, current_player_color)
            for action in possible_actions_tmp:
                tmp_state1 = deepcopy(tmp_state)
                if action != 65:
                    tmp_state = ReversiEnv.make_place(tmp_state, action, 0)
                score = self.calculate_min_max_action(tmp_state, depth+1, True)
                tmp_state = tmp_state1
                best_score = min(score, best_score)

            return best_score



    def best_action(self, game_state):
        import operator
        import itertools
        best_score = -9999
        tmp_state = game_state
        possible_actions_tmp = deepcopy(ReversiEnv.get_possible_actions(tmp_state, 1))
        action_score_dict = dict()
        for action in possible_actions_tmp:
            tmp_state1 = deepcopy(tmp_state)
            if action != 65:
                tmp_state = ReversiEnv.make_place(tmp_state, action, 1)

            score = self.calculate_min_max_action(tmp_state, 0, False)
            tmp_state = tmp_state1
            action_score_dict.update({action : score})
            if score > best_score:
                best_score = score
                b_action = action
            action_score_dict_sorted = sorted(action_score_dict.items(), key=lambda kv: kv[1])
            k_best_actions = action_score_dict_sorted[-2::]
            b_action = k_best_actions[self.np_random.randint(len(k_best_actions))][0]
        return b_action







