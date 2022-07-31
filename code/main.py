#Import modules
import time
import numpy as np

from qiea import QiEA
from greedy import Greedy
from hungarian import Hungarian
from simplex import Simplex

from auxiliar import distribution

#Initialize Solvers
def qiea_solver(n,m,a,theta,agents,samples,epochs,freq):
    initial = time.time()
    solver = QiEA(n,m,a,theta,agents,samples)
    counter = 0
    
    for epoch in range(epochs):
        solver.collapse()        
        solver.compare()
        solver.update()

        if counter == freq*epochs:
            counter=0
            solver.migration()
        else:
            counter+=1

    Q,x,b,best = solver.results()
    timer = time.time()-initial
    return best, solver.reward(best).sum(1).item(), timer

def greedy_solver(m,a,agents,samples):
    initial = time.time()
    solver = Greedy(m,a,agents,samples)
    solution, reward = solver.solve()
    timer = time.time()-initial
    return solution, reward, timer

def hungarian_solver(m,a,agents,samples):
    initial = time.time()
    solver = Hungarian(m,a,agents,samples)
    solution, reward = solver.solve()
    timer = time.time()-initial
    return solution, reward, timer

def simplex_solver(m,a,agents,samples):
    initial = time.time()
    solver = Simplex(m,a,agents,samples)
    solution, reward = solver.solve()
    timer = time.time()-initial
    return solution, reward, timer

#EXECUTE ALGORITHMS
def init_state(situation,m,a,is_agent=True): #situation,is_agent,m,a
    agents = distribution(situation,True,m,a)
    samples = distribution(situation,False,m,a)
    return agents,samples

def test(agents,samples,n,m,a,theta,epochs,freq):
    b,rew,time = qiea_solver(n,m,a,theta,agents,samples,epochs,freq)
    b2,rew2,time2 = greedy_solver(m,a,agents,samples)
    b3,rew3,time3 = hungarian_solver(m,a,agents,samples)
    b4,rew4,time4 = simplex_solver(m,a,agents,samples)
    print('MODEL','REWARD','TIME')
    print('QiEA:',rew,time/epochs)
    print('Greedy:',rew2,time2)
    print('Hungarian:',rew3,time3)
    print('Simplex:',rew4,time4)


situation = 'random'
n = 5
m = 10
a = 5
theta = 0.01*np.pi
epochs = 100
freq = 0.25

agents,samples = init_state(situation,m,a)
test(agents,samples,n,m,a,theta,epochs,freq)