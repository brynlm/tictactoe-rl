#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from IPython.display import clear_output
# import pandas as pd


# In[688]:


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
            
    def reset(self):
        self.board = np.full((3,3),0)
    
    def isGameOver(self, c):
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
    


# In[695]:


class Player():
    #REQUIRES: c is either 0 or 1 
    def __init__(self, b, c):
        self.b = b
        self.c = c
        
    def setPiece(self, coord):
        self.b.setPiece(coord,self.c)


# In[697]:


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
        
        


# In[ ]:


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

