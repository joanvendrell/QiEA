#Import module
import numpy as np
import torch
import random
from auxiliar import *

class QiEA(torch.nn.Module):
    def __init__(self,n,m,a,theta,agents,samples):
        self.n = n  #Populations
        self.m = m  #possible positions
        self.a = a  #agents
        self.theta = torch.tensor(theta) #Angle
        #Q(t) = {q_1,...,q_n}   --> n,a,2,m 
        #q_i = (1/2 |...| 1/2,  | |
        #       1/2 |...| 1/2,  a |
        #       1/2 |...| 1/2), | m
        #       (............)    |
        self.Q = torch.ones(self.n,self.a,2,self.m)*torch.sqrt(torch.tensor(1/2))
        self.x = torch.zeros(self.n,self.a,self.m)
        self.b = initializate(torch.zeros(self.n,self.a,self.m),self.m,self.a)
        #Reward-Cost
        self.agents = agents   #sigma of each agent
        self.samples = samples #sigma of each centroid
        if self.m>=self.a:
           self.best = torch.nn.functional.one_hot(torch.arange(0,self.a),num_classes=self.m).view(1,self.a,self.m)
        else:
           self.best = initializate(torch.zeros(1,self.a,self.m),self.m,self.a)
        #Info
        self.counter=0
        self.binary01 = torch.tensor([[0],[1]])
        self.binary10 = torch.tensor([[1],[0]])
                
    def collapse(self):
        self.x = torch.zeros(self.n,self.a,self.m)
        
        av_k_idx,counter = list(range(self.n)),[0 for i in range(self.n)]
        av_a_idx,av_c_idx = [list(range(self.a)) for i in range(self.n)],[list(range(self.m)) for i in range(self.n)]
        while sum(counter)!=self.a*self.n:
            k = np.random.choice(av_k_idx)
            i = np.random.choice(av_a_idx[k])
            j = np.random.choice(av_c_idx[k])
            if random.random()<self.Q[k,i,1,j].abs()**2:
                self.x[k][i][j] = 1
                counter[k]+=1
                av_a_idx[k].pop(av_a_idx[k].index(i))
                av_c_idx[k].pop(av_c_idx[k].index(j))
                if counter[k] == self.a:
                   av_k_idx.pop(av_k_idx.index(k))  
        """     
        for k in range(self.n):
            av_a_idx,av_c_idx,counter = list(range(self.a)),list(range(self.m)),0
            while counter!=min(self.a,self.m):
                i = np.random.choice(av_a_idx)
                j = np.random.choice(av_c_idx)
                if random.random()<self.Q[k,i,1,j].abs()**2:
                    self.x[k][i][j] = 1
                    counter+=1
                    av_a_idx.pop(av_a_idx.index(i))
                    av_c_idx.pop(av_c_idx.index(j))
        """
        """
        for k in range(self.n):
            av_c_idx = list(range(self.m))
            for i in range(self.a):
                j = np.random.choice(av_c_idx)
                if random.random()<self.Q[k,i,1,j].abs()**2:
                    self.x[k][i][j] = 1
                    av_c_idx.pop(av_c_idx.index(j))
        """
                
    def reward(self,input):
        # pi_k*[ln(pi_k/w_s)+0.5(ln(s_x*s_y/s_kx*s_ky)+(s_kx*s_y+s_x*s_ky)/(s_x*s_y)-2)]
        s_x = (input*self.agents.x).sum(2)
        s_y = (input*self.agents.y).sum(2)
        w_s = (input*self.agents.w).sum(2)
        #print('REWARD QIEA:',input.shape,self.samples.x.shape)
        s_kx = (input*self.samples.x).sum(2)
        s_ky = (input*self.samples.y).sum(2)
        pi_k = (input*self.samples.pi).sum(2)
        
        y=(pi_k*(torch.log(pi_k/w_s)+0.5*(torch.log((s_x*s_y)/(s_kx*s_ky))+(s_kx*s_y+s_x*s_ky)/(s_x*s_y)-2))).abs()
        return y
        
    def update(self):
        q_reward = self.reward(self.x) 
        b_reward = self.reward(self.b)
        #get angle matrix
        theta=(self.b-self.x)*(((b_reward-q_reward)>0).view(self.n,self.a,1)*torch.ones(self.n,self.a,self.m))*self.theta
        #check quadrant
        alpha = torch.atan(((self.Q*torch.flip(self.Q,[2,3]))*torch.tensor([[0],[1]]))).sum(2)
        a1 = torch.where(0<=alpha,1,0)*torch.where(alpha<=torch.tensor(np.pi/2),1,0)
        a2 = torch.where(torch.tensor(np.pi)<=alpha,1,0)*torch.where(alpha<torch.tensor(3*np.pi/2),1,0)
        a3 = torch.where(torch.tensor(np.pi/2)<=alpha,1,0)*torch.where(alpha<torch.tensor(np.pi),1,0)
        a4 = torch.where(torch.tensor(3*np.pi/2)<=alpha,1,0)*torch.where(alpha<=torch.tensor(2*np.pi),1,0)
        quadrants = a1 + a2 - a3 - a4
        #rotate
        theta = theta * quadrants
        cos = torch.cos(theta).view(self.n,self.a,1,self.m); sin = torch.sin(theta).view(self.n,self.a,1,self.m)
        b1 = (self.Q*cos)*torch.tensor([[1],[0]]) + (torch.flip(self.Q,[2,3])*(-sin))*torch.tensor([[1],[0]])
        b2 = torch.flip(self.Q,[2,3])*(sin)*torch.tensor([[0],[1]]) + self.Q*cos*torch.tensor([[0],[1]])
        self.Q = b1 + b2
        
    def compare(self):
        q_reward = self.reward(self.x).sum(1)
        b_reward = self.reward(self.b).sum(1)
        for k in range(self.n):
            if q_reward[k]<=b_reward[k]:
                self.b[k] = self.x[k]
        
    def migration(self):
        best_global = self.reward(self.best).sum(1)
        best_local = self.b[torch.argmax(self.reward(self.b).sum(1))].view(1,self.a,self.m)
        #Global
        if best_global > self.reward(best_local).sum(1):
            self.best = best_local
        #Local
        for k in range(self.n):
            r = random.random()
            if r<=0.5:
                self.b[k] = self.best

    def results(self):
        return self.Q,self.x,self.b,self.best
