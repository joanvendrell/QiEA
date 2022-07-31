#Import modules
import torch
import scipy.optimize

#Algorithm
class Simplex(torch.nn.Module):
    def __init__(self,m,a,agents,samples):
        self.m = m  #possible positions
        self.a = a  #agents
        #Reward-Cost
        self.agents = agents   #sigma of each agent
        self.samples = samples #sigma of each centroid
        #Conditions
        self.boundary=(0,1)
        self.Aeq = torch.zeros(self.a,self.a*self.m)
        self.beq = torch.ones(self.a)
        self.Aub = torch.zeros(self.a,self.a*self.m)
        self.bub = torch.ones(self.a)
        #fill
        original = 0
        for i in range(self.a):
            for e in range(self.m):
                self.Aeq[i][original+e] = 1
            original+=self.m
        original = 0
        for i in range(self.a):
            idx = original
            for e in range(self.a):
                self.Aub[i][idx+self.m*e] = 1
            original += 1
        
    def reward(self,input):
        # pi_k*[ln(pi_k/w_s)+0.5(ln(s_x*s_y/s_kx*s_ky)+(s_kx*s_y+s_x*s_ky)/(s_x*s_y)-2)]
        s_x = (input*self.agents.x).sum(2)
        s_y = (input*self.agents.y).sum(2)
        w_s = (input*self.agents.w).sum(2)
        s_kx = (input*self.samples.x).sum(2)
        s_ky = (input*self.samples.y).sum(2)
        pi_k = (input*self.samples.pi).sum(2)
        
        y=(pi_k*(torch.log(pi_k/w_s)+0.5*(torch.log((s_x*s_y)/(s_kx*s_ky))+(s_kx*s_y+s_x*s_ky)/(s_x*s_y)-2))).abs()
        return y

    def solve(self):
        solution = torch.zeros(self.a,self.m)
        #First compute all possibities
        possibilities = torch.zeros(self.a,self.m)
        for i in range(self.a):
            sx,sy,ws = self.agents.x[i][0],self.agents.y[i][0],self.agents.w[i][0]
            for j in range(self.m):
                kx,ky,kpi = self.samples.x[j],self.samples.y[j],self.samples.pi[j]
                #compute reward
                r=kpi*(torch.log(kpi/ws)+0.5*(torch.log((sx*sy)/(kx*ky))+(kx*sy+sx*ky)/(sx*sy)-2))
                possibilities[i][j] = r.abs()
        #Apply Simplex
        possibilities = possibilities.view(self.a*self.m)
        y = scipy.optimize.linprog(possibilities,A_ub=self.Aub,b_ub=self.bub,
                                   A_eq=self.Aeq,b_eq=self.beq,
                                   bounds=self.boundary,method='simplex')
        solution = torch.tensor(y.x).view(1,self.a,self.m)
        return solution, self.reward(solution).sum(1).item()
