import os
import pickle

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import pingouin as pg
from scipy.ndimage import gaussian_filter1d

from utils.agent import *
from utils.analyze import *
from utils.viz import viz
viz.get_style()

# set up path 
path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f'{path}/figures'): os.mkdir(f'{path}/figures')

feedback_types = ['gain', 'loss']
patient_groups = ['HC', 'PAT']
block_types    = ['sta', 'vol']
policies       = ['EU', 'MO', 'HA']
dpi = 300
width = .85

# fit performance 
def quantTable(agents= ['MOS', 'FLR', 'RP']):
    crs = {}
    
    for m in agents:
        n_params = eval(m).n_params
        nll, aic = 0, 0 
        fname = f'{path}/simulations/exp1data/{m}/sim-exp1data-map-idx0.csv'
        data  = pd.read_csv(fname)
        nll   = data.groupby(by=['sub_id'])['logLike'].sum().mean()
        aic   = 2*nll + 2*n_params
        bic   = 2*nll + n_params*np.log(data.groupby(by=['sub_id'])['logLike'].sum().shape[0])
        crs[m] = {'nll': nll, 'aic': aic, 'bic': bic}

    for m in agents:
        print(f'{m}({eval(m).n_params}) nll: {crs[m]["nll"]:.3f}, aic: {crs[m]["aic"]:.3f}, bic: {crs[m]["bic"]:.3f}')

def PrefxGroup(data):
    '''Decision preference X patient group
    '''

    tars = ['l1', 'l2', 'l3']
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data = data.groupby(by=['sub_id', 'is_PAT', 'feedback_type', 'b_type']
                    )[tars].mean().reset_index()
    
    fig, axs = plt.subplots(1, 3, figsize=(11, 4), sharey=True, sharex=True)
    t_test(data, 'is_PAT==False', 'is_PAT==True', tar=tars)

    for i, t in enumerate(tars):
        ax = axs[i]
        sns.boxplot(x='is_PAT', y=t, data=data, 
                width=width,
                palette=viz.PurplePairs, ax=ax)

        p =0 
        for box in ax.patches:
            if box.__class__.__name__ == 'PathPatch':
                box.set_edgecolor(viz.PurplePairs[p%2])
                box.set_facecolor('white')
                for k in range(6*p, 6*(p+1)):
                    ax.lines[k].set_color(viz.PurplePairs[p%2])
                p += 1
        sns.stripplot(x='is_PAT', y=t, data=data, 
                        jitter=True, dodge=True, marker='o', size=7,
                        palette=viz.PurplePairs, alpha=0.5,
                        ax=ax)
        ax.set_ylim([-5, 5])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['HC', 'PAT'])
        ax.set_xlabel('')
        ax.set_ylabel('  \n ')
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{path}/figures/PrefxGroup.png', dpi=dpi)

def PrefxSyndrome(data):

    tars  = ['g', 'g', 'g']
    preds = ['l1', 'l2', 'l3']
    predlabels = [r'$log(W_{EU})$',r'$log(W_{MO})$',r'$log(W_{HA})$']
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data = data.groupby(by=['sub_id'])[['g', 'l1', 'l2', 'l3']].mean().reset_index()

    nr, nc = 1, int(len(tars))
    fig, axs = plt.subplots(nr, nc, figsize=(11, 3.5), sharey=True)
    for i, (pred, tar) in enumerate(zip(preds, tars)):
        xmin, xmax = data[pred].values.min()-.4, data[pred].values.max()+.4
        x = data[pred]
        y = data[tar]
        print(f'\n-----{tar}:')
        print(pg.corr(x.values, y.values).round(3))
        lm = pg.linear_regression(x, y)
        print(tabulate(lm.round(3), headers='keys', tablefmt='fancy_grid'))
        
        ax  = axs[i]
        x = np.linspace(xmin, xmax, 100)
        sns.scatterplot(x=pred, y=tar, data=data, s=100, alpha=.5, 
                            color=viz.Palette2[i], ax=ax)
        ax.set_xlim([xmin, xmax]) # for the regression predictor 
        sns.regplot(x=pred, y=tar, data=data, truncate=False,
                        color=viz.Palette2[i], scatter=False, ax=ax)
        ax.set_ylabel('General factor (A.U.)')
        ax.set_xlabel(predlabels[i])
        ax.set_box_aspect(1)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
 
    plt.tight_layout()
    plt.savefig(f'{path}/figures/PrefxSyndrome.png', dpi=300)

def PrefxEnv(data):
    '''Decision preference X Env volatility
    '''

    tars = ['l1', 'l2', 'l3']
    data = data.groupby(by=['sub_id', 'b_type'])[tars].mean().reset_index()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    t_test(data, 'b_type=="sta"', 'b_type=="vol"', tar=tars)

    new_data = []
    for t in tars:
        temp_data = data.loc[:, ['b_type', t]].rename(columns={t: 'weights'})
        temp_data['params'] = t
        new_data.append(temp_data)
    new_data = pd.concat(new_data, axis=0)

    sns.boxplot(x='params', y='weights', hue='b_type', data=new_data, 
                palette=viz.BluePairs, ax=ax)
    p =0 
    for box in ax.patches:
        if box.__class__.__name__ == 'PathPatch':
            box.set_edgecolor(viz.BluePairs[p%2])
            box.set_facecolor('white')
            for k in range(6*p, 6*(p+1)):
                ax.lines[k].set_color(viz.BluePairs[p%2])
            p += 1
    sns.stripplot(x='params', y='weights', hue='b_type', data=new_data, 
                    jitter=True, dodge=True, marker='o', size=7,
                    palette=viz.BluePairs, alpha=0.5,
                    ax=ax)
    ax.set_ylim([-5, 5])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r'$log(W_{EU})$',r'$log(W_{MO})$',r'$log(W_{HA})$'])
    ax.set_xlabel('')
    ax.set_ylabel('Decision \nPreference (A.U.)')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Stable', 'Volatile'], 
                bbox_to_anchor=(1.3, .6), loc='right')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{path}/figures/PrefxEnv.png', dpi=dpi)

def LRxEnv(data):
    '''learning rate X Env volatility
    '''

    tars = ['log_alpha']
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data = data.groupby(by=['sub_id', 'is_PAT', 'feedback_type', 'b_type']
                        )[tars].mean().reset_index()
    
    fig, ax = plt.subplots(1, 1, figsize=(3.9, 4))
    t_test(data, 'b_type=="sta"', 'b_type=="vol"', tar=tars)
    sns.boxplot(x='b_type', y='log_alpha', data=data, 
                width=width,
                palette=viz.BluePairs, ax=ax)
    p =0 
    for box in ax.patches:
        if box.__class__.__name__ == 'PathPatch':
            box.set_edgecolor(viz.BluePairs[p%2])
            box.set_facecolor('white')
            for k in range(6*p, 6*(p+1)):
                ax.lines[k].set_color(viz.BluePairs[p%2])
            p += 1
    sns.stripplot(x='b_type', y='log_alpha', data=data, 
                    jitter=True, dodge=True, marker='o', size=7,
                    palette=viz.BluePairs, alpha=0.5,
                    ax=ax)
    ax.set_ylim([-2.4, 0])
    ax.set_xlabel('')
    ax.set_ylabel(' ')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Stable', 'Volatile'])

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{path}/figures/LRxEnv.png', dpi=dpi)

def LRxGroup(data):
    '''learning rate X patient group
    '''

    tars = ['log_alpha']
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data = data.groupby(by=['sub_id', 'is_PAT', 'feedback_type', 'b_type']
                    )[tars].mean().reset_index()
    
    fig, ax = plt.subplots(1, 1, figsize=(3.9, 4))
    t_test(data, 'is_PAT==False', 'is_PAT==True', tar=tars)

    sns.boxplot(x='is_PAT', y='log_alpha', data=data, 
                palette=viz.PurplePairs, ax=ax)
    p =0 
    for box in ax.patches:
        if box.__class__.__name__ == 'PathPatch':
            box.set_edgecolor(viz.PurplePairs[p%2])
            box.set_facecolor('white')
            for k in range(6*p, 6*(p+1)):
                ax.lines[k].set_color(viz.PurplePairs[p%2])
            p += 1
    sns.stripplot(x='is_PAT', y='log_alpha', data=data, 
                    jitter=True, dodge=True, marker='o', size=7,
                    palette=viz.PurplePairs, alpha=0.5,
                    ax=ax)
    ax.set_ylim([-2.4, 0])
    ax.set_xlabel('')
    ax.set_ylabel(' ')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['HC', 'PAT'])

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{path}/figures/LRxGroup.png', dpi=dpi)

def PrefdiffxGroup(data):
    '''Preference difference X patient group
    '''

    tars = ['l1', 'l2', 'l3']
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data = data.groupby(by=['sub_id', 'is_PAT', 'b_type', 'feedback_type']
                    )[tars].mean().reset_index()
    for i, t in enumerate(tars):
        print(f'# ------------------ {t} ----------------- #')
        print(pg.anova(dv=t, between=['is_PAT', 'b_type', 
                    'feedback_type'], data=data).round(3))
    
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    data = data.groupby(by=['sub_id', 'is_PAT', 'b_type']
                    )[tars].mean().reset_index()
    for i, t in enumerate(tars):
        
        ax = axs[i]
        sns.boxplot(x='is_PAT', y=t, hue='b_type', data=data, 
                width=width,
                palette=viz.BluePairs, ax=ax)
       
        p =0 
        for box in ax.patches:
            if box.__class__.__name__ == 'PathPatch':
                box.set_edgecolor(viz.BluePairs[p%2])
                box.set_facecolor('white')
                for k in range(6*p, 6*(p+1)):
                    ax.lines[k].set_color(viz.BluePairs[p%2])
                p += 1
        sns.stripplot(x='is_PAT', y=t, hue='b_type', data=data, 
                        jitter=True, dodge=True, marker='o', size=7,
                        palette=viz.BluePairs, alpha=0.5,
                        ax=ax)
        
        ax.set_xlabel('')
        ax.set_ylabel('  \n ')
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['HC', 'PAT'])
        ax.get_legend().remove()

    plt.tight_layout()
    plt.savefig(f'{path}/figures/PrefdiffxGroup.png', dpi=dpi)


def Stategy_Ada():

    fname = f'{path}/simulations/exp1data/MOS/simsubj-exp1data-sta_first-AVG.csv'
    data = pd.read_csv(fname)

    data = data.groupby(by=['trials'])[['l1_effect', 'l2_effect', 'l3_effect']].mean()
    psi  = np.zeros([180])
    psi[:90]     = .7
    psi[90:110]  = .2
    psi[110:130] = .8
    psi[130:150] = .2
    psi[150:170] = .8
    psi[170:180] = .2

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax = ax 
    labels = ['EU', 'MO', 'HA']
    for i in range(3):
        sns.lineplot(x='trials', y=f'l{int(i)+1}_effect', lw=3, 
                    data=data, color=viz.Palette2[i], label=labels[i])
    sns.lineplot(x=np.arange(180), y=psi, color='k', ls='--')
    ax.set_ylabel('Prob. of choosing \nthe left stimulus')
    ax.set_xlabel('Trials')
    ax.set_ylim([-.1, 1.1])
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{path}/figures/effect.png', dpi=dpi)

def Pi_Ada():

    psi  = np.zeros([180])
    psi[:90]     = .7
    psi[90:110]  = .2
    psi[110:130] = .8
    psi[130:150] = .2
    psi[150:170] = .8
    psi[170:180] = .2

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax = ax 
    for i, g in enumerate(['HC', 'PAT']):
        fname = f'{path}/simulations/exp1data/MOS/simsubj-exp1data-sta_first-{g}.csv'
        data = pd.read_csv(fname)
        data = data.groupby(by=['trials'])[['pi']].mean()
        sns.lineplot(x='trials', y=f'pi', lw=3, 
                    data=data, color=viz.PurplePairs[i], label=g)
    sns.lineplot(x=np.arange(180), y=psi, color='k', ls='--')
    ax.set_ylabel('Prob. of choosing \nthe left stimulus')
    ax.set_xlabel('Trials')
    ax.legend()
    ax.set_ylim([-.1, 1.1])
    plt.tight_layout()
    plt.savefig(f'{path}/figures/effect2.png', dpi=dpi)

def humanAda(mode):

    with open(f'data/{mode}_exp1data.pkl', 'rb')as handle:
        data = pickle.load(handle)

    cases = {'sta0.7-vol0.2':[], 'sta0.3-vol0.8':[]}

    for subj in data.keys():
        datum = data[subj][0]
        ind = {}
        for btype in ['sta', 'vol']:
            ind[btype] = list(range(90)) if datum.loc[0, 'b_type'] == btype\
                            else list(range(90, 180)) 

        ## stable 
        sel_data = datum.query('b_type=="sta"'
                ).groupby(by='state').count()['trial']
        datum.loc[ind['sta'], 'p0'] = np.round(sel_data[0] / 90, 2)

        ## volatile 
        idx1, idx2 = ind['vol'][0], ind['vol'][0]+19
        n = datum.loc[idx1:idx2].groupby(by='state').count()['trial'][0] / 20
        datum.loc[ind['vol'], 'p0'] = [.2]*20+[.8]*20+[.2]*20+[.8]*20+[.2]*10 if n==.2 else\
                                    [.8]*20+[.2]*20+[.8]*20+[.2]*20+[.8]*10 
        cond = f'{datum.loc[0, "b_type"]}{datum.loc[0, "p0"]}-{datum.loc[90, "b_type"]}{datum.loc[90, "p0"]}'
        if cond in cases.keys():
            cases[cond].append(datum) 
             
    plt.figure(figsize=(10, 4))
            
    cs = ['group=="HC"', 'group!="HC"']
    sel_data = pd.concat(cases['sta0.7-vol0.2']).reset_index()
    sns.lineplot(x='trial', y='p0', data=sel_data, ls='--', color='k')
    lbs = ['HC', 'PAT']
    for i, c in enumerate(cs):
        sel_data = pd.concat(cases['sta0.7-vol0.2']).query(c).reset_index()
        sel_data['humanAct'] = sel_data['humanAct'].apply(
            lambda x: x if mode=='loss' else 1-x)
        sel_data2 = pd.concat(cases['sta0.3-vol0.8']).query(c).reset_index()
        sel_data2['humanAct'] = sel_data2['humanAct'].apply(
            lambda x: 1-x if mode=='loss' else x)
        sdata = pd.concat([sel_data, sel_data2],axis=0, ignore_index=True)
        print(f'{lbs[i]}: {sdata.shape[0]/180}')
        a = sdata.groupby(by='trial')[['humanAct']].mean().reset_index().rolling(5
                        ).mean().values[5-1:]
        b = gaussian_filter1d(a[:,1], sigma=2)
        c = a[:177, 0]
        sdata = pd.DataFrame(np.vstack([c.reshape([-1]), b]).T, columns=['trial', 'humanAct'])
        # sdata = pd.DataFrame(sdata.groupby(by='trial')[['humanAct']].mean().reset_index().rolling(5
        #                 ).mean().values[5-1:], columns=['trial', 'humanAct'])
        sns.lineplot(x='trial', y='humanAct', data=sdata, color=viz.PurplePairs[i], ci=0, label=lbs[i])
    plt.legend()
    plt.ylim([-.1, 1.1])
    plt.xlabel('Trials')
    plt.ylabel('Prob. of choosing \nthe left stimulus')
    plt.tight_layout()
    plt.savefig(f'{path}/figures/human_{mode}.png', dpi=300)


if __name__ == '__main__':

    quantTable()

    ## parameters analyses
    pivot_table = build_pivot_table('map', agent='MOS', min_q=.01, max_q=.99)

    # PrefxGroup(pivot_table)
    # PrefxSyndrome(pivot_table)
    # PrefxEnv(pivot_table)
    # LRxEnv(pivot_table)
    # LRxGroup(pivot_table)
    # PrefdiffxGroup(pivot_table)
    # Stategy_Ada()
    # Pi_Ada()
    humanAda('gain')
    humanAda('loss')