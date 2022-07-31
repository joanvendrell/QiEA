#Import modules
import time
import numpy as np
import pandas as pd
import pickle

from qiea import QiEA
from greedy import Greedy
from hungarian import Hungarian
from simplex import Simplex

from auxiliar import distribution
from plot import *

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

#####-------TEST-------#####
#load environment
data = pd.read_csv('data/hyperparameters.csv')
env = []
with (open("data/environments.pkl", "rb")) as openfile:
    while True:
        try:
            env.append(pickle.load(openfile))
        except EOFError:
            break
env = env[0]

#for each Case Study, select the accurate/fast hyperparameters
accurate_cases = {x: {'cost': None, 'timer': None,
                  'parameters': {'population': None, 'theta': None, 'epochs': None, 'migration': None}} for x in list(env.keys())}
fast_cases = {x: {'cost': None, 'timer': None,
              'parameters': {'population': None, 'theta': None, 'epochs': None, 'migration': None}} for x in list(env.keys())}

for i in range(len(data)):
    d = data.loc[i]
    situation, cost, timer = d.Situation, d.Cost, d.Time
    if accurate_cases[situation]['cost'] == None:
        accurate_cases[situation]['cost'] = cost
        accurate_cases[situation]['timer'] = timer
        accurate_cases[situation]['parameters']['population'] = d.Population
        accurate_cases[situation]['parameters']['theta'] = d.Theta
        accurate_cases[situation]['parameters']['epochs'] = d.Epochs
        accurate_cases[situation]['parameters']['migration'] = d.Migration
        
    if fast_cases[situation]['cost'] == None:
        fast_cases[situation]['cost'] = cost
        fast_cases[situation]['timer'] = timer
        fast_cases[situation]['parameters']['population'] = d.Population
        fast_cases[situation]['parameters']['theta'] = d.Theta
        fast_cases[situation]['parameters']['epochs'] = d.Epochs
        fast_cases[situation]['parameters']['migration'] = d.Migration
    else:
        if accurate_cases[situation]['cost'] > cost:
            accurate_cases[situation]['cost'] = cost
            accurate_cases[situation]['timer'] = timer
            accurate_cases[situation]['parameters']['population'] = d.Population
            accurate_cases[situation]['parameters']['theta'] = d.Theta
            accurate_cases[situation]['parameters']['epochs'] = d.Epochs
            accurate_cases[situation]['parameters']['migration'] = d.Migration

        if fast_cases[situation]['timer'] > timer:
            fast_cases[situation]['cost'] = cost
            fast_cases[situation]['timer'] = timer
            fast_cases[situation]['parameters']['population'] = d.Population
            fast_cases[situation]['parameters']['theta'] = d.Theta
            fast_cases[situation]['parameters']['epochs'] = d.Epochs
            fast_cases[situation]['parameters']['migration'] = d.Migration

print('****HYPERPARAMETERS****')
print('--Accurate QiEA')
print('Case', 'Population', 'Epochs', 'Theta', 'Migration Frequency')
for case in accurate_cases:
    print(case,accurate_cases[case]['parameters']['population'],accurate_cases[case]['parameters']['epochs'],
          accurate_cases[case]['parameters']['theta'],accurate_cases[case]['parameters']['migration'])
print('--Speed QiEA')
print('Case', 'Population', 'Epochs', 'Theta', 'Migration Frequency')
for case in accurate_cases:
    print(case,fast_cases[case]['parameters']['population'],fast_cases[case]['parameters']['epochs'],
          fast_cases[case]['parameters']['theta'],fast_cases[case]['parameters']['migration'])


#compare with other models
cases = {x: {'Greedy':{'cost': None, 'timer': None},
             'Hungarian':{'cost': None, 'timer': None},
             'Simplex':{'cost': None, 'timer': None},
             'AQiEA':{'cost': None, 'timer': None},
             'FQiEA':{'cost': None, 'timer': None}} for x in list(env.keys())}

for case in cases:
    agents,targets,m,a = env[case]['agent'],env[case]['samples'],env[case]['m'],env[case]['a']
    rew1,time1 = accurate_cases[case]['cost'], accurate_cases[case]['timer']
    rew2,time2 = fast_cases[case]['cost'], fast_cases[case]['timer']
    _,rew3,time3 = greedy_solver(m,a,agents,targets)
    _,rew4,time4 = hungarian_solver(m,a,agents,targets)
    _,rew5,time5 = simplex_solver(m,a,agents,targets)
    #save
    cases[case]['AQiEA']['cost'] = rew1; cases[case]['AQiEA']['timer'] = time1
    cases[case]['FQiEA']['cost'] = rew2; cases[case]['FQiEA']['timer'] = time2
    cases[case]['Greedy']['cost'] = rew3; cases[case]['Greedy']['timer'] = time3
    cases[case]['Hungarian']['cost'] = rew4; cases[case]['Hungarian']['timer'] = time4
    cases[case]['Simplex']['cost'] = rew5; cases[case]['Simplex']['timer'] = time5

print('****RESULTS****')
print('--Accuracy')
print('Case','Greedy','Hungarian','Simplex','AQiEA','FQiEA')
for case in cases:
    print(case,cases[case]['Greedy']['cost'],cases[case]['Hungarian']['cost'],cases[case]['Simplex']['cost'],
          cases[case]['AQiEA']['cost'],cases[case]['FQiEA']['cost'])
print('--Speed')
print('Case','Greedy','Hungarian','Simplex','AQiEA','FQiEA')
for case in cases:
    print(case,round(cases[case]['Greedy']['timer']*1000,3),round(cases[case]['Hungarian']['timer']*1000,3),
          round(cases[case]['Simplex']['timer']*1000,3),round(cases[case]['AQiEA']['timer']*1000,3),
          round(cases[case]['FQiEA']['timer']*1000,3))

#Plot results
plot_results(cases)
for case in cases:
    plot_gird_search(case,env,data,accurate_cases,fast_cases)