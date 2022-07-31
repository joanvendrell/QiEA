#Import modules
import time
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from qiea import QiEA

from auxiliar import init_state, save_environment

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

#Hyperparameters
situations = {('unbalanced_big_big'):[100,90],
              'unbalanced_small_small':[8,2],
              'unbalanced_big_small':[100,2],
              'unbalanced_small_big':[8,5],
              'balanced':[10,10]}
n_populations = [2,5,10]
angles = [0.0025*np.pi,0.01*np.pi,0.05*np.pi]
epochs = [10,20,50]
migrations = [0.25,0.5,0.75]
    

#Grid Search
columns = ['Situation','Population','Clusters','Agents','Theta','Epochs','Migration','Time','Cost']
df = pd.DataFrame(columns=columns)
environment = {}
counter=0

for e,situation in enumerate(situations):
    m,a = situations[situation]
    agents,samples = init_state('random',m,a)
    #save_environment(agents,samples,'data/services_'+str(m)+'_'+str(a)+'.pt','data/targets_'+str(m)+'_'+str(a)+'.pt')
    environment[situation]={}
    environment[situation]['agent'] = agents #{'x':agents.x,'y':agents.y,'w':agents.w}
    environment[situation]['samples'] = samples #{'x':samples.x,'y':samples.y,'pi':samples.pi}
    environment[situation]['m'] = m
    environment[situation]['a'] = a
    
    for f,n in enumerate(n_populations):
        for i,theta in enumerate(angles):
            for j,epoch in enumerate(epochs):
                for k,freq in enumerate(migrations):
                    try:
                        print('Epoch '+str(counter)+' of '+
                            str(len(situations)*len(n_populations)*len(angles)*len(epochs)*len(migrations)))
                        print(situation,n,m,a,theta,epoch,freq)
                        sol,rew,timer = qiea_solver(n,m,a,theta,agents,samples,epoch,freq)
                        #Save info
                        df.loc[counter] = [situation,n,m,a,theta,epoch,freq,timer/epoch,rew]
                        environment[situation][counter] = sol
                        counter+=1
                    except AssertionError:
                        pass
#Save data
df.to_csv('data/hyperparameters.csv')
f = open("data/environments.pkl","wb")
pickle.dump(environment,f)
