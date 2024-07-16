#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import tensorflow as tf
import utils
from tictactoegame import TicTacToe as Board
from tictactoegame import Player
from tictactoegame import RandomMoves
import matplotlib.pyplot as plt

# currently, the data generation strategy is inefficient since everything is going to be re-evaluated during the forward pass.


class Agent(Player):
    def __init__(self,c,policy=None):
        if policy is None:
            policy = tf.keras.Sequential([tf.keras.layers.Dense(64,activation='relu') for i in range(2)]+
                                        [tf.keras.layers.Dense(9,activation='softmax')])
        super().__init__(c)
        self.policy = policy # <-- policy is a Keras model taking 3x3 array to 1x9 array of probabilities
        self.policy.build((None,18))
    
    @tf.function
    def __call__(self, st): # <-- st has shape (batch, 18)
        return self.policy(st)
    
    def makeMove(self, board):
        actions = self.getAvailPosn(board)
        if len(actions)==0:
            return False
        st = utils.board_to_features(board.board)
        st = st.T
        pA = tf.squeeze(self.policy(st))
        action = np.random.choice(9, p=pA.numpy())
        coord = divmod(action,3)
        if board.board[coord[0],coord[1]] != 0: 
            board.board[coord[0],coord[1]]=9
            return action
        board.setPiece(coord, self.c)
        return action

class Session():
    def __init__(self,X=None,O=None,lr=1e-3,gamma=0.8, t_it=10):
        self.history={'wr':[], 'lr':[], 'dr':[], 'ir':[]}
        self.t_it=t_it
        self.it=0 
        
        self.lr=lr
        self.gamma=gamma
        self.discount=tf.constant([gamma**(gamma-t-1) for t in range(8)],dtype=tf.float32)
        self.Xs = X if X is not None else RandomMoves(1)
        self.Os = O if O is not None else RandomMoves(-1)
        self.board = Board()
        self.opt=tf.keras.optimizers.Adam(-1*lr)
        self.hasGenerator=False
        self.hasDataset=False
        
    def get_data_gen(self):
        if self.hasGenerator==True:
            return self.data_gen
        def data_generator():
            while True:
                S,A=[],[]
                for st,at in self:
                    S.append(st)
                    A.append(at)
                S=tf.concat(S,axis=0)
                A=tf.stack(A,axis=0)
                yield (S,A),self.get_reward()
        self.data_gen=data_generator
        self.hasGenerator=True
        return self.data_gen
    
    def getDataset(self):
        if self.hasDataset==True:
            return self.dataset
        self.dataset=tf.data.Dataset.from_generator(self.get_data_gen(),output_signature=((tf.TensorSpec(shape=[None,18],dtype=tf.int32),
                                                                        tf.TensorSpec(shape=[None],dtype=tf.int32)),tf.TensorSpec(shape=None)))
        self.hasDataset=True
        return self.dataset
    
    @tf.function
    def get_pg(self,S,A,r,discount):
        A = tf.one_hot(A,depth=9)
        idxs = tf.where(A)
        V = discount*r
        with tf.GradientTape() as tape:
            P = self.Xs(S)
            P = tf.gather_nd(P,idxs)
            logP = tf.math.log(P)
            logP = tf.reduce_sum(logP*V)
        grads = tape.gradient(logP, self.Xs.policy.variables)
        return grads
    
    def train_step(self,batch_size=10):
        R=tf.constant([0.0])
        aveG=[]
        for (S,A),r in self.getDataset().take(batch_size): # <-- maybe this can be stacked and passed through model all at once
            length=tf.shape(S)[0]
            grads=self.get_pg(S,A,r,self.discount[:length])
            if len(aveG) == 0:
                aveG = grads
            else:
                for i in range(len(grads)):   # <-- for computing expected policy gradient
                    aveG[i] += grads[i]
            R=tf.concat([R,r[tf.newaxis]],axis=0)
        aveG=[grad / batch_size for grad in aveG]
        self.opt.apply_gradients(list(zip(aveG,self.Xs.policy.variables)))
        return tf.reduce_mean(R[1:])
    
    def train_loop(self,epochs,batch_size):
        epoch=0
        aveR=[]
        while epoch < epochs:
            print(epoch)
            aveR.append(self.train_step(batch_size))
            epoch+=1
        return aveR
        
    def get_reward(self):
        if self.board.isGameOver(1):
            return 1
        if self.board.isGameOver(-1):
            return -1
        if 9 in self.board.board:
            return -1
        if len(self.board.getPosns(0))==0:
            return 0
        return 0
        
    def __iter__(self): # <-- only record Xs moves, for now.
        self.board.reset()
        if self.it % self.t_it == 0:
            for key in self.history:
                self.history[key].append(0)
        self.it+=1
        return self
    
    def __next__(self):
        if self.board.isGameOver(1):
            self.history['wr'][-1] += 1/self.t_it
            raise StopIteration
        if self.board.isGameOver(-1):
            self.history['lr'][-1] += 1/self.t_it
            raise StopIteration
        if len(self.board.getPosns(9))!=0:
            self.history['ir'][-1] += 1/self.t_it
            raise StopIteration
        elif len(self.board.getPosns(0)) == 0:
            self.history['dr'][-1] += 1/self.t_it
            raise StopIteration
        else:
            st,at = utils.board_to_features(self.board.board).T, self.Xs.makeMove(self.board)
            if self.board.isGameOver(1) or len(self.board.getPosns(9))!=0:
                return st,at
            self.Os.makeMove(self.board)
            return st,at