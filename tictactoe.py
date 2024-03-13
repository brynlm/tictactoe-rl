#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from IPython.display import clear_output
import pandas as pd


# In[2]:


class TicTacToe():
    def __init__(self):
        self.board = np.full((3,3), 0)
    
    # coord = (x,y)
    def setPiece(self,coord,c):
        row = coord[0]
        col = coord[1]
        if self.board[row,col] == 0:
            self.board[row,col] = c
            return self.board
        else:
            return "err"
#             print("invalid move")
            
    def reset(self):
        self.board = np.full((3,3),0)
    
    def isGameOver(self, c):
        if "err" in self.board:
            return false
        return (self.Horizontal(c) or self.Vertical(c) or self.Diagonal(c))
    
    def getPosns(self, c):
        try:
            positions = np.where(self.board == c)
            positions = [i for i in zip(positions[0], positions[1])]
        except:
            print('np.where = ', np.where(self.board == c))
            
        return positions
    
    def Horizontal(self, c):
        for i in range(3):
            k = 0
            for j in self.board[i,:]:
                if j == c:
                    k+=1
                else: 
                    break
            if k == 3:
                return True
        return False
    
    def Vertical(self, c):
        for i in range(3):
            k = 0
            for j in self.board[:,i]:
                if j == c:
                    k+=1
                else: 
                    break
            if k == 3:
                return True
        return False
    
    def Diagonal(self, c):
        k = 0
        for i in range(3):
            if self.board[i,i] == c:
                k+=1
            else:
                break
        if k==3:
            return True
        else:
            for i in range(3):
                if self.board[i, 2-i] == c:
                    k+=1
                else:
                    break
            return (k==3)
    


# In[3]:


class Player():
    #REQUIRES: c is either 0 or 1 
    def __init__(self, b, c):
        self.b = b
        self.c = c
        
    def setPiece(self, coord):
        self.b.setPiece(coord,self.c)


# In[4]:


class RandomMoves(Player):
    def __init__(self, b, c):
        super().__init__(b, c)
        self.myTurn = False
        
    def setTurn(self, toggle):
        self.myTurn = toggle
    
    def getAvailPosn(self):
        return self.b.getPosns(0)
    
    def makeMove(self):
        actions = self.getAvailPosn()
#         i = np.random.randint(len(actions))
        i = np.random.default_rng().integers(len(actions))
#         self.setPiece(actions[i])
#         super().setPiece(actions[i])
        new = super().setPiece(actions[i])
        return new
        
        
#         try:
#             i = np.random.randint(len(actions))
#             self.setPiece(actions[i])
#         except:
#             pass
#             print(actions)
        


# In[702]:


board = TicTacToe() 
print(board.board)
board.setPiece((1,1), 1)
print(board.setPiece((0,2), -1))
# board.getPosns(0)
np.where(board.board == 0)


# In[703]:


p1 = RandomMoves(board, 1)
p2 = RandomMoves(board, -1)
game = Game(board, p1, p2)

for i in game:
    print(game.b.board)


# In[ ]:


for i in game:
    print(p1._pr)
    print(game.b.board)
    


# In[568]:


z = np.zeros((3,3,2))
l = [(1,2,0), (1,2,1), (1,1,0)]
for i in l:
    x,y,s = i
    z[x,y,s] += 1
print(z)


# In[ ]:


help(np.random.randint)


# In[509]:


it = iter(game)


# In[ ]:


next(it)
print(game._t1)
print(game.b.board)


# In[ ]:



p.makeMove() #for some reason 2 moves are made even when called just once?
             #^SOLVED: some kind of complication due to inheritance; resolved by changing call to
             #         setPiece to super().setPiece. 
# p.makeMove()
print(p.b.board)


# In[691]:


class Game():
    def __init__(self, b, p1, p2):
        self.b = b
        self.p1 = p1
        self.p2 = p2
    
    def __iter__(self):
        self._t1 = True
        self.p1.b = self.b #resetting player 1 board without changing other attributes
        self.p2.b = self.b
#         self.p1.setTurn(True)
        return self
    
    def __next__(self):
        if self.b.isGameOver(self.p1.c):
#             print('p1 won')
            self._winner = self.p1
            if isinstance(self.p1, Agent):
                S_f = self.p1.getState()
                last_x = self.p1._last_s[0]
                last_y = self.p1._last_s[1]
                self.p1._pr_next_s[last_x, last_y, S_f[0], S_f[1]] += 1
        
            raise StopIteration
            
        elif self.b.isGameOver(self.p2.c):
#             print('p2 won')
            self._winner = self.p2
            if isinstance(self.p1, Agent):
                S_f = self.p1.getState()
                last_x = self.p1._last_s[0]
                last_y = self.p1._last_s[1]
                self.p1._pr_next_s[last_x, last_y, S_f[0], S_f[1]] += 1
            
            raise StopIteration
        elif len(self.b.getPosns(0)) == 0:
#             print('draw')
            self._winner = None
            if isinstance(self.p1, Agent):
                S_f = self.p1.getState()
                last_x = self.p1._last_s[0]
                last_y = self.p1._last_s[1]
                self.p1._pr_next_s[last_x, last_y, S_f[0], S_f[1]] += 1
            
            raise StopIteration
        else:
#             print(self._t1)
#             print(self.b.getPosns(' '))
#             print(self.b.board)
            if self._t1:
                self.p1.makeMove()
                self.b = self.p1.b  #  <--- adding this line seemed to fix the "valueError" coming from np.random.randint
                                    #  assumingly because player and game TicTacToe objects and distinct
                self._t1 = not self._t1
                return self._t1
                
#                 return self.b
            else:
                self.p2.makeMove()
                self.b = self.p2.b 
                self._t1 = not self._t1
                return self._t1
#                 return self.b


# In[675]:


#since first player always can force a draw, should I set a negative reward for draws as well? 
reward = 0
wins   = 0
losses = 0
draws  = 0
player1 = Agent(board, 1)
player2 = RandomMoves(board, -1)
board = TicTacToe()


# action_count = np.zeros((3,3,2))
for n in range(100):
    steps = 0
#     player1 = Agent(board, 'x')
# #     player1 = RandomMoves(board,'x')
#     player2 = RandomMoves(board, 'o')
    board = TicTacToe()
    game = Game(board, player1, player2)
    
#     print(n)
    
    for i in game:
        steps+=1
#         print(steps)
        pass
        
    if game._winner == player1:
        reward +=1
        wins +=1
#         game.b.reset()
    elif game._winner == player2:
        reward -=1
        losses +=1
#         game.b.reset()
    else:
        draws += 1
#         game.b.reset()
        continue
    
def get_prob(arr):
    a = arr
    for i in range(4):
        for j in range(4):
            total = np.sum(arr[i,j,:])
            if a[i,j,0] != 0:
                a[i,j,0] = a[i,j,0] / total
#             arr[i,j,0] = arr[i,j,0] / total
            if a[i,j,1] != 0:
                a[i,j,1] = a[i,j,1] / total
    return a
            
print('reward = ', reward)
print('wins   = ', wins)
print('losses = ', losses)
print('draws  = ', draws)
print(player1._pr)
print(get_prob(player1._pr))
# print(player1._pr_next_s)

# was getting "valueError: 0 <= high" for np.random.randint, 


# In[670]:


#example code for extracting number of wins losses and draws from State data, given a number of games played
loss_states = player1._pr_next_s[:,:,:,3]
losses = player1._pr_next_s[:,2,:,3]
wins = player1._pr_next_s[2,:,3,:]

print('# losses = ', np.sum(losses))
print('# wins   = ', np.sum(wins))
print('# draws  = ', 100 - np.sum(wins) - np.sum(losses))

print(player1._pr_next_s[0,0,:,:])
print(player1._pr_next_s[:,:,0,0])


# In[668]:


# EFFECTS: returns probability of State s occuring given previous state s0.
def s_Pr_sr(s,s0,S_count):
    s0_count = S_count[s0[0],s0[1],:,:]
    total = np.sum(s0_count)
    if total == 0:
        return 0
    else:
        prob_s = s0_count[s[0],s[1]] / total
        return prob_s
    
# EFFECTS: returns reward corresponding to one of the three terminal states (win, lose, draw)
def reward(s):
    if s[0] == 3:
        r = 1
    elif s[1] == 3:
        r = -1
    else:
        r = 0
    return r

# EFFECTS: "base case" policy evaluation (state-value function)
def v0(s_count, acc=0):
    values = np.zeros((4,4))
    states = [(i,j) for i in range(4) for j in range(4)]
    for i in states:
        values[i[0],i[1]] = np.sum([s_Pr_sr(m, i, s_count)*reward(m) for m in states])
#         print(values[i[0],i[1]])
    return values

# EFFECTS: iterative policy evaluation (state-value function)
#^ im not sure if I did this right, since it was my first attempt.

def value(s_count, acc=v0(player1._pr_next_s), step=0):
    if step < 50:
        gamma = 0.80
        values = acc
        states = [(i,j) for i in range(4) for j in range(4)]
        for s in states:
            values[s[0], s[1]] = np.sum([s_Pr_sr(m, s, s_count)*(reward(m)+gamma*values[m[0], m[1]]) for m in states])
            
        return value(s_count, acc=values, step= step+1)
    else:
        return acc

print(v0(player1._pr_next_s))
print(value(player1._pr_next_s))


# In[639]:


l_x = list(range(3))
l_y = list(range(3))
r2 = [(i,j) for i in l_x for j in l_y]
# print(r2)

S_count = player1._pr_next_s
win_states = [(3,i) for i in range(4)]
one_away_states = [(2,i) for i in range(4)]

for j in one_away_states:
    for i in win_states:
        prob = s_Pr_sr(i, j, S_count)
        print('Prob(',i,'|', j, ') = ', prob)
        
l = [s_Pr_sr(i, j, S_count) for i in win_states for j in one_away_states]
print(l)
prob_win = np.sum([s_Pr_sr(i, j, S_count) for i in win_states for j in one_away_states])

print(prob_win)


# In[674]:


#NOTE** Eventually, I want to rework this project by using deep-learning to associate board configuration to States,
#       instead of manually defining the set of States and Actions (example, I want it to learn what it means to "block"
#       rather than have to hard code it)

#Define set of (non-terminal) States S:
# 1) two-away
# 2) one-away
# 3) opponent is two-away
# 4) opponent is one-away
# 5) 1 X [3, 4]
# 6) 2 X [3, 4]

#Define set of Actions A:
# 1) extend
# 2) branch (branch is a subset of extend, may not need this distinction)
# 3) block  (block can intersect with extend)

class Agent(Player):
    def __init__(self, b, c):
        super().__init__(b,c)
        self._pr = np.zeros((4,4,2)) #<--before policy evaluation
        self._last_s = None
#         self._pr_next_s = np.zeros((4,4,12)) #<-- count of S_t -> S_t'. Third axis (axis=2) represents S_t'
        self._pr_next_s = np.zeros((4,4,4,4))
        self._history = []
        
    # EFFECTS: makes move and records state-action pair, and state-s' pairs.
    def makeMove(self):
        actions = self.getAvailPosn()

        i = np.random.randint(len(actions))
        extendable = self.getExtend() # <-- needs to be called BEFORE setPiece, so looking at set of positions
        blockable  = self.getBlock()  #     before board changed.

        S_t = self.getState()
        x = S_t[0]
        y = S_t[1]
        
        if self._last_s is not None:
            last_x = self._last_s[0]
            last_y = self._last_s[1]
#             self._pr_next_s[last_x, last_y, 3*x + y] += 1 #<-- this flattened indexing approach may not be suitable
            self._pr_next_s[last_x, last_y, x, y] += 1
    
        self._last_s = tuple(S_t)
        
        super().setPiece(actions[i])
        
    
#         print('S_t = ', S_t)
#         print(extendable)
        if actions[i] in extendable:
            self._pr[x,y,0] += 1
        if actions[i] in blockable:
            self._pr[x,y,1] += 1
    
    def getAvailPosn(self):
        return self.b.getPosns(0)
    
    # EFFECTS: scans board and returns positions where chain can be extended
    #          a position is extendable if at least one of the row, column, or diagonal it shares
    #          contains only x's or blanks.
    def getExtend(self):
        empty = self.getAvailPosn()
        extendable = filter(lambda coord : self.isExtendable(coord, self.c), empty)
        return list(extendable)
      
    # EFFECTS: returns True if position is blockable extendable for opponent (otherwise, it is already blocked)
    def getBlock(self):
        empty = self.getAvailPosn()
        blockable = filter(lambda coord : self.isExtendable(coord, -1), empty)
        return list(blockable)
        
    # REQUIRES: element at coord is empty
    def isExtendable(self, coord, c):
        x = coord[1]
        y = coord[0]
        empty_row = True
        empty_col = True
        empty_dia0 = True
        empty_dia1 = True
        
        for i in self.b.board[y,:]:
                if i == 0 or i == c:
                    continue
                else:
                    empty_row = False
                    break
        for i in self.b.board[:,x]:
            if i == 0 or i == c:
                continue
            else:
                empty_col = False
                break
                    
        for i in range(3):
            if self.b.board[i,i] == 0 or self.b.board[i,i] == c:
                continue
            else:
                empty_dia = False
                break

        for i in range(3):
            if self.b.board[i,2-i] == 0 or self.b.board[i,2-1] == c:
                continue
            else:
                empty_dia = False
                break
        
        if ((x == 0 | x == 2) & (y == 0 | y == 2)):
            if y == 0:
                return (empty_row | empty_col | empty_dia0)
            else:
                return (empty_row | empty_col | empty_dia1)
            
        if x == 1 & y == 1:
            return (empty_row | empty_col | empty_dia0 | empty_dia1)
        
        else:
            return (empty_row | empty_col)

    # EFFECTS: scans board and returns State S_t as a 2x1 matrix.
    def getState(self):
        def get_largest(l, acc=0):
            if len(l) == 0:
                return acc
            else:
                if l[0] >= acc:
                    return get_largest(l[1:],l[0])
                else:
                    return get_largest(l[1:],acc)
                
        S_t = np.array([get_largest([self.HorizontalState(self.c), 
                                     self.VerticalState(self.c), 
                                     self.DiagonalState(self.c)]),
                        get_largest([self.HorizontalState(-1), 
                                     self.VerticalState(-1), 
                                     self.DiagonalState(-1)])])
        return S_t
    
    # EFFECTS: returns length of longest unblocked horizontal chain (if blocked, returns 0)
    def HorizontalState(self, c):
        longest = 0
        for i in range(3):
            r = 0
            for j in range(3):
                if self.b.board[i,j] == c:
                    r +=1
                elif self.b.board[i,j] == 0:
                    pass
                else:
                    r = 0
                    break
            if r > longest:
                longest = r
        return longest
                
    # EFFECTS: returns length of longest unblocked vertical chain (if blocked, returns 0)
    def VerticalState(self, c):
        longest = 0
        for i in range(3):
            v = 0
            for j in range(3):
                if self.b.board[j,i] == c:
                    v +=1
                elif self.b.board[j,i] == 0:
                    pass
                else:
                    v = 0
                    break
            if v > longest:
                longest = v
        return longest
                
    # EFFECTS: returns length of longest unblocked diagonal chain (if blocked, returns 0)
    def DiagonalState(self, c):
        d1 = 0
        d2 = 0
        for i in range(3):
            if self.b.board[i,i] == c:
                d1 += 1
            elif self.b.board[i,i] == 0:
                pass
            else:
                d1 = 0
                break
        for i in range(3):
            if self.b.board[i,2-i] == c:
                d2 += 1
            elif self.b.board[i,2-i] == 0:
                pass
            else:
                d2 = 0
                break
           
        if d1 >= d2:
            return d1
        else:
            return d2
        
            

        
        


# In[645]:


t3 = TicTacToe()
agent = Agent(t3, 'x')
player = Player(t3, 'o')

agent.setPiece((1,0))
# agent.setPiece((1,1))
player.setPiece((1,2))
player.setPiece((2,0))
player.setPiece((2,2))
player.setPiece((0,1))
agent.setPiece((0,0))
agent.setPiece((0,2))
agent.setPiece((2,1))

empty = agent.getAvailPosn()

result = (agent.b.board == 'x') | (agent.b.board == ' ')

np.where(result) # <-- how to get coordinates


print(agent.b.board)
# print(agent.HorizontalState('x'))
# print(agent.VerticalState('x'))
# print(agent.DiagonalState('x'))
# print(agent.isExtendable((0,2)))
print(agent.getState())
print(empty)
print(agent.getExtend())
print(agent.getBlock())
print(agent.isExtendable((0,2), 'o'))
block = list(filter(lambda coord : agent.isExtendable(coord,'o'), empty))
print(block)


# In[303]:


def get_largest(l, acc=0):
           if len(l) == 0:
               return acc
           else:
               if l[0] >= acc:
                   return get_largest(l[1:],l[0])
               else:
                   return get_largest(l[1:],acc)
               
l = [7,8,5,4,6]
get_largest(l)


# In[ ]:




