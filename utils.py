#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import tensorflow as tf
from tictactoegame import TicTacToe as Board


# In[5]:


def game_step(policy, start=False):
    board = Board()
    if isinstance(start, type(np.full((3,3),0))):
        board.board=np.copy(start)
    while True:
        if 9 in board.board:
            yield np.copy(board.board)
            break
        if board.isGameOver(1) or board.isGameOver(-1):
            yield np.copy(board.board)
            break
        S_t = board_to_features(np.copy(board.board))
#         A_t = get_action(tf.reshape(policy(S_t), (9,1)).numpy())
        A_t = get_action(tf.squeeze(policy(tf.reshape(S_t, (1,18)))).numpy())
        yield (A_t, np.copy(board.board))
        if board.board[A_t[0], A_t[1]] != 0:
            board.board[A_t[0], A_t[1]] = 9
            continue
        board.setPiece(A_t, 1)
        actions = board.getPosns(0)
        if len(actions) == 0:
            yield np.copy(board.board)
            break
        i = np.random.default_rng().integers(len(actions)) #<--- rework this part to allow games between models
        board.setPiece(actions[i], -1)
    return
     
def gen_episode(policy, state=np.full((3,3), 0)):
    episode = []
    for i in game_step(policy, state):
        episode.append(i)
        
    return episode

def board_to_features(board):
    board = tf.reshape(board, [9,1])
    p1_posn = np.asarray((board==1))
    p2_posn = np.asarray((board==-1))
    features = np.concatenate([p1_posn, p2_posn], axis=0)
    return features


# Could it make a difference by representing features as 9x2 array instead of 18x1?
def one_hot(board):
    board = tf.reshape(board, [9,1])
    p1 = np.asarray((board==1))
    p2 = np.asarray((board==-1))
    features = np.concatenate([p1,p2], axis=1)
    return features
    

def get_action(probs, batch_size=None):
    probs /= probs.sum()
    p = np.copy(probs)
#     print(p)
    a = np.random.choice(9,batch_size, p=p)
    return divmod(a,3)
#     return a

#     rng = np.random.default_rng()
#     a = rng.random()
#     low = 0
#     for i in range(len(action_probs)):
#         if action_probs[i] == 0:
#             continue
#         if (low <= a and a < action_probs[i] + low):
#             if i==0:
#                 return (0,0)
#             else:
#                 return divmod(i,3)
#         else:
#             low += action_probs[i]

def reward_fn(state):
    board = Board()
    board.board = state
    if board.isGameOver(1):
        return 10
    if board.isGameOver(-1):
        return -1
    if 9 in board.board:
        return -1
    return 0

def average_win_rate(policy, num_trials=100):
    wins = 0
    for i in range(num_trials):
        ep = gen_episode(policy)
        R = reward_fn(ep[-1])
        if R > 0:
            wins += 1
    return wins / num_trials

def discourage_illegal(state):
    board = Board()
    board.board = state
    if 9 in board.board:
        return -1
    return 1

def discounted_return(episode, gamma, reward_fn):
    ret = 0
    for i in range(len(episode)-1):
        ret += reward_fn(episode[i][1])*gamma**(len(episode)-1-i)
    return ret + reward_fn(episode[-1])


"""
Need to think about and potentially re-implement the expected return calculation.
Currently, the episode lengths for the "state" value estimates
"""
def expected_return(state, policy,reward_fn, gamma, return_fn=False, num_trials=100):
    returns = 0
    if return_fn == False:
        for i in range(num_trials):
            ep = gen_episode(policy, state)
            for j in ep[:-1]: 
                returns += reward_fn(j[1])
            returns += reward_fn(ep[-1])
    else:
        for i in range(num_trials):
            ep = gen_episode(policy, state)
            returns += return_fn(ep, gamma, reward_fn)
    return returns/num_trials

# def expected_return(state, policy, reward_fn=reward_fn, num_trials=100):
#     reward = 0
#     for i in range(num_trials):
#         ep = gen_episode(policy, state)[-1]
#         reward += reward_fn(ep)
#     return reward/num_trials

def state_dist(policy, num_trials, state=np.full((3,3), 0)):
    illegal_moves = 0
    lost = 0
    won = 0
    draws = 0
    for i in range(num_trials):
        board = Board()
        board.board = gen_episode(policy, state)[-1]
        if 9 in board.board:
            illegal_moves += 1
        elif board.isGameOver(1):
            won +=1
        elif board.isGameOver(-1):
            lost +=1
        else:
            draws += 1
    illegal_moves = illegal_moves/num_trials
    won = won/num_trials
    lost = lost/num_trials
    draws = draws/num_trials
    
    return illegal_moves, lost, won, draws

