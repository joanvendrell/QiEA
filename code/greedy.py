#Import modules
import torch

#Algorithm
class Greedy(torch.nn.Module):
    def __init__(self,m,a,agents,samples):
        self.m = m  #possible positions
        self.a = a  #agents
        #Reward-Cost
        self.agents = agents   #sigma of each agent
        self.samples = samples #sigma of each centroid

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
        #walk through possibilities and get the max reward
        agents = list(range(self.a))
        clusters = list(range(self.m))
        while solution.sum()<min(self.a,self.m):
            values = possibilities.min(1).values.tolist()
            index = possibilities.min(1).indices.tolist()
            #choose
            max_idx = values.index(min(values))
            index_agent = agents[max_idx]
            index_clust = clusters[index[max_idx]]
            #update solution
            solution[index_agent][index_clust] = 1
            #udpate parameters
            possibilities = possibilities.tolist()
            agents.pop(max_idx)
            possibilities.pop(max_idx)
            clusters.pop(index[max_idx])
            for i in range(len(possibilities)):
                possibilities[i].pop(index[max_idx])
            possibilities = torch.tensor(possibilities)
        solution = solution.view(1,self.a,self.m)
        return solution, self.reward(solution).sum(1).item()
