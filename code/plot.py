import matplotlib.pyplot as plt
import matplotlib as mpl
import math

mpl.use("pgf")
mpl.rcParams.update({
"pgf.texsystem": "pdflatex",
"font.family": "serif",
"text.usetex": True,
"pgf.rcfonts": False,
"figure.dpi": 150
})

def plot_gird_search(case,env,data,accurate_cases,fast_cases):
    idxs = [list(env[case].keys())[x] for x in range(len(list(env[case].keys()))) if type(list(env[case].keys())[x]) == int]
    timer,cost,min_t_idx,min_c_idx = [],[],0,0
    for idx in idxs:
        d = data.iloc[idx]
        assert d.Situation == case
        timer.append(d.Time)
        cost.append(d.Cost)
        if d.Time == fast_cases[case]['timer']:
            min_t_idx = len(timer)-1
        if d.Cost == accurate_cases[case]['cost']:
            min_c_idx = len(cost)-1
    #Save Cost
    plt.plot(timer, color = 'black')
    plt.plot(timer,'b.') 
    plt.plot(min_t_idx,fast_cases[case]['timer'],'r.') #Fastest Combination
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.savefig('data/plots/cost_grid_search_'+case+'.pgf')
    plt.close()
    #Save Time
    plt.plot(cost, color = 'black')
    plt.plot(cost,'b.') 
    plt.plot(min_c_idx,accurate_cases[case]['cost'],'r.') #More Accurate Combination
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.savefig('data/plots/speed_grid_search_'+case+'.pgf')
    plt.close()

def plot_results(cases):
    x = ['Greedy','Hungarian','Simplex','AQiEA','FQiEA']
    for case in cases:
        if case != 'unbalanced_big_big':
            colors = ['aquamarine','turquoise','lightseagreen','darkorange','orange']
        else:
            colors = ['aquamarine','turquoise','indianred','darkorange','orange']
        #--accuracy
        y = [cases[case][idx]['cost'] if not math.isnan(cases[case][idx]['cost']) else 0 for idx in x]
        plt.bar(x,y,color=colors)
        plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
        #plot reference lines
        max_y = [max(y) for i in range(len(y))]
        color = 'red' #colors[y.index(max(y))]
        plt.plot(max_y,color=color)
        #
        min_y = [min(y) for i in range(len(y))]
        color = 'green' #colors[y.index(min(y))]
        plt.plot(min_y,color=color)
        #
        acc_y = [y[3] for i in range(len(y))]
        color = colors[3]
        plt.plot(acc_y,color=color,linestyle='dashed')
        #
        plt.savefig('data/plots/accuracy_'+case+'.pgf')
        plt.close()
        #--speed
        y = [cases[case][idx]['timer'] if cases[case][idx]['timer']<=300 else 0 for idx in x]
        plt.bar(x,y,color=colors)
        plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
        #plot reference lines
        max_y = [max(y) for i in range(len(y))]
        color = 'red' #colors[y.index(max(y))]
        plt.plot(max_y,color=color)
        #
        min_y = [min(y) for i in range(len(y))]
        color = 'green' #colors[y.index(min(y))]
        plt.plot(min_y,color=color)
        #
        fast_y = [y[4] for i in range(len(y))]
        color = colors[4]
        plt.plot(fast_y,color=color,linestyle='dashed')
        #
        plt.savefig('data/plots/speed_'+case+'.pgf')
        plt.close()
