#Import module
import torch
import random

#Function to intializate random collapsed states
def initializate(input,m,a):
    for k in range(len(input)):
        agents = list(range(a))
        clusters = list(range(m))
        if m>a:
           while len(agents)!=0:
              i,j = random.choice(agents),random.choice(clusters)
              input[k][i][j] = 1
              agents.pop(agents.index(i))
              clusters.pop(clusters.index(j))
        else:
           while len(clusters)!=0:
              i,j = random.choice(agents),random.choice(clusters)
              input[k][i][j] = 1
              agents.pop(agents.index(i))
              clusters.pop(clusters.index(j))
    return input

#Function to initializate agents/clusters distributions
class distribution():
    """
    Situations:
       Situation 1 --> 12 agents, 12 clusters as YC paper
       Situation 2 --> 7 agents, 12 clusters
       Situation 3 --> 5 agents, 100 clusters (random)
       Situation 4 --> 100 agents, 100 clusters (random)
    """
    def __init__(self,situation,is_agent,m,a):
        self.is_agent = is_agent
        self.m = m
        self.a = a
        if situation == 'equal_YC':
            assert self.m == 12 and self.a == 12
            if self.is_agent:
                #           1    2    3    4    5    6
                #           7    8    9   10   11   12
                self.x = [[30],[30],[80],[70],[30],[60],
                          [50],[30],[40],[10],[20],[20]]
                self.y = [[30],[15],[30],[25],[30],[40],
                          [20],[70],[15],[30],[40],[50]]
                self.w = [[0.2],[0.15],[0.2],[0.15],[0.3],[0.2],
                          [0.3],[0.25],[0.2],[0.5],[0.1],[0.2]]
              
                self.x = torch.tensor(self.x) * torch.ones(self.a,self.m)
                self.y = torch.tensor(self.y) * torch.ones(self.a,self.m)
                self.w = torch.tensor(self.w) * torch.ones(self.a,self.m)

                self.optimal = [[0,0,0,1,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0,0,1],
                                [0,0,0,0,0,0,0,1,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0,1,0],
                                [0,0,0,0,0,1,0,0,0,0,0,0],
                                [0,1,0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,1,0,0,0],
                                [0,0,0,0,0,0,0,0,0,1,0,0],
                                [0,0,0,0,0,0,0,1,0,0,0,0],
                                [0,0,1,0,0,0,0,0,0,0,0,0],
                                [1,0,0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,1,0,0,0,0,0,0,0]]
                self.sol = [4,12,8,11,6,2,9,10,7,3,1,5]
            else:
                self.x = torch.tensor([70,20,30,20,60,30,40,10,50,80,30,30])
                self.y = torch.tensor([25,50,70,40,40,15,15,30,20,30,30,30])
                self.pi = torch.tensor([0.15,0.2,0.25,0.1,0.2,0.15,0.2,0.5,0.3,0.2,0.2,0.3])

        elif situation == 'no_equal_YC':
            assert self.m == 12 and self.a == 7
            if self.is_agent:
                #           1    2    3    4    5    6   7
                self.x = [[30],[30],[80],[70],[30],[60],[50]]
                self.y = [[30],[15],[30],[25],[30],[40],[20]]
                self.w = [[0.2],[0.15],[0.2],[0.15],[0.3],[0.2],[0.3]]
                self.x = torch.tensor(self.x) * torch.ones(self.a,self.m)
                self.y = torch.tensor(self.y) * torch.ones(self.a,self.m)
                self.w = torch.tensor(self.w) * torch.ones(self.a,self.m)

                self.optimal = [[0,0,1,0,0,0,0,0,0,0,0,0],
                                [1,0,0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,1,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,1,0,0,0,0,0],
                                [0,0,0,1,0,0,0,0,0,0,0,0],
                                [0,1,0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,1,0,0,0,0,0,0]]
                self.sol = [3,1,5,7,4,2,6]
            else:
                self.x = torch.tensor([80,30,30,50,70,30,60, 80, 40, 100, 80, 30])
                self.y = torch.tensor([30,30,30,20,25,15,40, 20, 30,  10, 25, 20])
                self.pi = torch.tensor([0.2,0.2,0.3,0.3,0.15,0.15,0.2, 0.1, 0.2, 0.15, 0.2, 0.3])
                
        elif situation == 'random':
            if self.is_agent:
                x,y,w = [],[],[]
                for i in range(self.a):
                    x.append(random.random()*100)
                    y.append(random.random()*100)
                    w.append(random.random())
                self.x = torch.tensor(x).view(self.a,1) * torch.ones(self.a,self.m)
                self.y = torch.tensor(y).view(self.a,1) * torch.ones(self.a,self.m)
                self.w = torch.tensor(w).view(self.a,1) * torch.ones(self.a,self.m)
                self.optimal = None
                self.sol = None
            else:
                x,y,w = [],[],[]
                for i in range(self.m):
                    x.append(random.random()*100)
                    y.append(random.random()*100)
                    w.append(random.random())
                self.x = torch.tensor(x)
                self.y = torch.tensor(y)
                self.pi = torch.tensor(w)
                self.pi = self.pi /sum(self.pi)

#Function to initializate states
def init_state(situation,m,a,is_agent=True): #situation,is_agent,m,a
    agents = distribution(situation,True,m,a)
    samples = distribution(situation,False,m,a)
    return agents,samples

#Function to create an environment
def save_environment(agents,samples,name_service,name_targets):
    print('Saving the service and targets...')
    torch.save(agents,name_service)
    torch.save(samples,name_targets)

#Function to read pre-saved services/targets
def load_environment(name_service,name_targets):
    agents = torch.load(name_service)
    samples = torch.load(name_targets)

    return agents,samples
