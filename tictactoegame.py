#!/usr/bin/env python
# coding: utf-8


import numpy as np
import tensorflow as tf

class TicTacToe():
    def __init__(self):
        self.board = np.full((3,3), 0)
    
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
    

# Abstract Player class 
class Player():
    # REQUIRES: c is either -1 or 1
    def __init__(self, c):
        self.c = c
    
    def getAvailPosn(self,board):
        return board.getPosns(0)
    
    def makeMove(self,board):
        pass
    
    
class RandomMoves(Player):
    
    def makeMove(self, board):
        actions = self.getAvailPosn(board)
        if len(actions)==0:
            return False
        i = np.random.default_rng().integers(len(actions))
        board.setPiece(actions[i],self.c)
        return actions[i]
    
