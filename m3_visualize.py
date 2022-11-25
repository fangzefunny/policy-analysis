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
from utils.bms import fit_bms
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

# ------------  Model comparison ------------- #

# fit performance 
def quantTable(agents= ['MOS', 'FLR', 'RP']):
    crs = {}
    n_param =[18, 15, 9, 6, 6, 3]
    ticks = [f'{m}({n})' for n, m in zip(n_param, agents)]

    for m in agents:
        n_params = eval(m).n_params
        nll, aic = 0, 0 
        fname = f'{path}/fits/exp1data/params-exp1data-{m}-bms-ind.csv'
        data  = pd.read_csv(fname)
        nll   = -data.loc[0, 'log_like']
        aic   = data.loc[0, 'aic']
        bic   = data.loc[0, 'bic']
        crs[m] = {'NLL': nll, 'AIC': aic, 'BIC': bic}

    for m in agents:
        print(f'{m}({eval(m).n_params}) nll: {crs[m]["NLL"]:.3f}, aic: {crs[m]["AIC"]:.3f}, bic: {crs[m]["BIC"]:.3f}')

    fig, axs = plt.subplots(3, 1, figsize=(6, 11))
    xx = list(range(len(agents)))
    for i, c in enumerate(['NLL', 'AIC', 'BIC']):
        cr = np.array([crs[m][c] for m in agents])
        cr -= cr.min()
        ax = axs[i]
        sns.barplot(x=xx, y=cr, palette=viz.Palette[:len(agents)], ax=ax)
        ax.set_xticks(xx)
        ax.set_xticklabels(ticks, rotation=45)
        ax.set_xlim([0-.8, len(agents)-1+.8])
        ax.set_ylabel(f'Delta {c}')
        plt.tight_layout()
        plt.savefig(f'{path}/figures/quant.png', dpi=300)


def show_bms(models = ['MOS', 'FLR', 'RP', 'MOS_fix', 'FLR_fix', 'RP_fix'], 
             n_param =[18, 15, 9, 6, 6, 3]):
    '''group-level bayesian model selection
    '''
    
    ticks = [f'{m}({n})' for n, m in zip(n_param, models)]
    fit_sub_info = []

    for i, m in enumerate(models):
        with open(f'fits/exp1data/fit_sub_info-{m}-bms.pkl', 'rb')as handle:
            fit_info = pickle.load(handle)
        # get the subject list 
        if i==0: subj_lst = fit_info.keys() 
        # get log post
        log_post = [fit_info[idx]['log_post'] for idx in subj_lst]
        bic      = [fit_info[idx]['bic'] for idx in subj_lst]
        h        = [fit_info[idx]['H'] for idx in subj_lst]
        n_param  = fit_info[list(subj_lst)[0]]['n_param']
        fit_sub_info.append({
            'log_post': log_post, 
            'bic': bic, 
            'n_param': n_param, 
            'H': h
        })

    bms_results = fit_bms(fit_sub_info)

    # show protected exceedence 
    _, ax = plt.subplots(1, 1, figsize=(5, 4))
    xx = list(range(len(models)))
    sns.barplot(x=xx, y=bms_results['pxp'], palette=viz.Palette[:len(models)], ax=ax)
    ax.set_xticks(xx)
    ax.set_xticklabels(ticks, rotation=45)
    ax.set_xlim([0-.8, len(models)-1+.8])
    ax.set_ylabel('PXP')
    plt.tight_layout()
    plt.savefig(f'{path}/figures/BMS.png', dpi=300)

# ---------- Model-based analysis ----------- #

def StylexConds(data, cond, fig_id):
    '''Decision style over different conditions

    Args:
        data: the preprocessed data 
        cond: the condition on the x axis
            - group: PAT, HC
            - volatility: stable, volatile
            - feedback: reward, aversive
        fig_id: figure id
    
    Outputs: 
        A bar plot 
    '''
    # prepare the inputs 
    tars = ['l1', 'l2', 'l3']
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data = data.groupby(by=['sub_id', 'group', 'feedback_type', 'b_type']
                    )[tars].mean().reset_index()

    # select condition
    if cond == 'group':
        varr, case1, case2 = 'group', 'HC', 'PAT'
        ticks = ['HC', 'PAT']
        colors = viz.PurplePairs
    elif cond == 'volatility':
        varr, case1, case2 = 'b_type', 'sta', 'vol'
        ticks = ['Stable', 'Volatile']
        colors = viz.BluePairs
    elif cond == 'feedback':
        varr, case1, case2 = 'feedback_type', 'gain', 'loss'
        ticks = ['Reward', 'Aversive']
        colors = viz.YellowPairs

    # show bar plot 
    fig, axs = plt.subplots(1, 3, figsize=(11, 4), sharey=True, sharex=True)
    t_test(data, f'{varr}=="{case1}"', f'{varr}=="{case2}"', tar=tars)

    for i, t in enumerate(tars):
        ax = axs[i]
        sns.boxplot(x=varr, y=t, data=data, 
                width=width,
                palette=colors, ax=ax)

        p =0 
        for box in ax.patches:
            if box.__class__.__name__ == 'PathPatch':
                box.set_edgecolor(colors[p%2])
                box.set_facecolor('white')
                for k in range(6*p, 6*(p+1)):
                    ax.lines[k].set_color(colors[p%2])
                p += 1
        sns.stripplot(x=varr, y=t, data=data, 
                        jitter=True, dodge=True, marker='o', size=7,
                        palette=colors, alpha=0.5,
                        ax=ax)
        ax.set_ylim([-5, 5])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(ticks)
        ax.set_xlabel('')
        ax.set_ylabel('  \n ')
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig{fig_id}_Stylex{cond}.png', dpi=300)

def StyleInter(data, cond, fig_id):
    '''Decision style over different conditions

    Args:
        data: the preprocessed data 
        cond: the condition on the x axis
            - group x volatitility
            - group x feedback: stable
            - volatility x feedback: reward
        fig_id: figure id
    
    Outputs: 
        A bar plot 
    '''
    # prepare the inputs 
    tars = ['l1', 'l2', 'l3']
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data = data.groupby(by=['sub_id', 'group', 'feedback_type', 'b_type']
                    )[tars].mean().reset_index()

    # select condition
    if cond == 'group-volatility':
        varr1, case11, case12 = 'group', 'HC', 'PAT'
        varr2, case21, case22 = 'b_type', 'sta', 'vol'
        ticks = ['HC', 'PAT']
        legs  = ['Stable', 'Volatile']
        colors = viz.BluePairs
    elif cond == 'group-feedback':
        varr1, case11, case12 = 'group', 'HC', 'PAT'
        varr2, case21, case22 = 'feedback_type', 'gain', 'loss'
        ticks = ['HC', 'PAT']
        legs  = ['Reward', 'Aversive']
        colors = viz.YellowPairs
    elif cond == 'volatility-feedback':
        varr1, case11, case12 = 'b_type', 'sta', 'vol'
        varr2, case21, case22 = 'feedback_type', 'gain', 'loss'
        ticks = ['Stable', 'Volatile']
        legs  = ['Reward', 'Aversive']
        colors = viz.YellowPairs

    # show bar plot 
    fig, axs = plt.subplots(1, 4, figsize=(14.5, 4), sharey=True, sharex=True)

    for i, t in enumerate(tars):
        ax = axs[i]
        sns.boxplot(x=varr1, y=t, data=data, hue=varr2,
                width=width,
                palette=colors, ax=ax)

        p =0 
        for box in ax.patches:
            if box.__class__.__name__ == 'PathPatch':
                box.set_edgecolor(colors[p%2])
                box.set_facecolor('white')
                for k in range(6*p, 6*(p+1)):
                    ax.lines[k].set_color(colors[p%2])
                p += 1
        sns.stripplot(x=varr1, y=t, data=data, hue=varr2, 
                        jitter=True, dodge=True, marker='o', size=7,
                        palette=colors, alpha=0.5, 
                        ax=ax)
        ax.set_ylim([-5, 5])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(ticks)
        ax.set_xlabel('')
        ax.set_ylabel('  \n ')
        ax.get_legend().remove()
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
    
    ax=axs[3]
    for i in range(2):
        sns.scatterplot(x=[0,1], y=[np.nan]*2, marker='s',
                s=120, color=colors[i], ax=ax)
    ax.set_axis_off()
    ax.legend(legs)

    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig{fig_id}_Stylex{cond}.png', dpi=300)

def StylexSyndrome(data, fig_id):
    '''Decision style over sydrome 

    Args:
        data: the preprocessed data 
        fig_id: figure id
    
    Outputs: 
        A bar plot 
    '''

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
    plt.savefig(f'{path}/figures/Fig{fig_id}_PrefxSyndrome.png', dpi=300)

def LRxConds(data, cond, fig_id):
    '''Learning rate over different conditions

    Args:
        data: the preprocessed data 
        cond: the condition on the x axis
            - group: PAT, HC
            - volatility: stable, volatile
            - feedback: reward, aversive
        fig_id: figure id
    
    Outputs: 
        A bar plot 
    '''
    # prepare for the fit
    tars = ['log_alpha']
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data = data.groupby(by=['sub_id', 'group', 'feedback_type', 'b_type']
                        )[tars].mean().reset_index()

    # select condition
    if cond == 'group':
        varr, case1, case2 = 'group', 'HC', 'PAT'
        ticks = ['HC', 'PAT']
        colors = viz.PurplePairs
    elif cond == 'volatility':
        varr, case1, case2 = 'b_type', 'sta', 'vol'
        ticks = ['Stable', 'Volatile']
        colors = viz.BluePairs
    elif cond == 'feedback':
        varr, case1, case2 = 'feedback_type', 'gain', 'loss'
        ticks = ['Reward', 'Aversive']
        colors = viz.YellowPairs
    
    t_test(data, f'{varr}=="{case1}"', f'{varr}=="{case2}"', tar=tars)

    fig, ax = plt.subplots(1, 1, figsize=(3.9, 4))
    sns.boxplot(x=varr, y='log_alpha', data=data, 
                width=width,
                palette=colors, ax=ax)
    p =0 
    for box in ax.patches:
        if box.__class__.__name__ == 'PathPatch':
            box.set_edgecolor(colors[p%2])
            box.set_facecolor('white')
            for k in range(6*p, 6*(p+1)):
                ax.lines[k].set_color(colors[p%2])
            p += 1
    sns.stripplot(x=varr, y='log_alpha', data=data, 
                    jitter=True, dodge=True, marker='o', size=7,
                    palette=colors, alpha=0.5,
                    ax=ax)
    ax.set_ylim([-2.4, 0])
    ax.set_xlabel('')
    ax.set_ylabel(' ')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(ticks)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig{fig_id}_LRx{cond}.png', dpi=dpi)

def StrategyAda(fig_id):

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
    plt.savefig(f'{path}/figures/Fig{fig_id}_StrategySim.png', dpi=dpi)

def PolicyAda(fig_id):

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
    plt.savefig(f'{path}/figures/Fig{fig_id}_PolicySim.png', dpi=dpi)

def HumanAda(mode, fig_id):

    psi  = np.zeros([180])
    psi[:90]     = .7
    psi[90:110]  = .2
    psi[110:130] = .8
    psi[130:150] = .2
    psi[150:170] = .8
    psi[170:180] = .2

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
    sns.lineplot(x=np.arange(180), y=psi, color='k', ls='--')
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
    plt.savefig(f'{path}/figures/Fig{fig_id}_HumanSim-{mode}.png', dpi=dpi)


if __name__ == '__main__':

    ## parameters analyses
    pivot_table = build_pivot_table('map', agent='MOS', min_q=.01, max_q=.99)
    pivot_table['group'] = pivot_table['group'].map(
                    {'HC': 'HC', 'MDD': 'PAT', 'GAD': 'PAT'})

    # --------- Main results --------- #

    # Table1: quantitative fit table 
    quantTable(['MOS', 'FLR', 'RP', 'MOS_fix', 'FLR_fix', 'RP_fix'])
    
    # Fig 2: Decision style effect
    StylexConds(pivot_table, 'group', fig_id='2A')   # Fig 2A
    StylexSyndrome(pivot_table, fig_id='2B')         # Fig 2B

    # Fig 3: Learning rate effect
    LRxConds(pivot_table, 'volatility', fig_id='3A') # Fig 3A
    LRxConds(pivot_table, 'group', fig_id='3B')      # Fig 3B

    # Fig 4: Understand the flexible behaviors
    HumanAda('loss', fig_id='4A')                    # Fig 4A
    PolicyAda(fig_id='4B')                           # Fig 4B
    StrategyAda(fig_id='4C')                         # Fig 4C

    # ------ Supplementary materials ------- #

    #Fig S1: Decision style effect 
    StylexConds(pivot_table, 'volatility', fig_id='S1A')   # Fig S1A
    StylexConds(pivot_table, 'feedback', fig_id='S1B')     # Fig S1b

    # Fig S2: Decision style interaction effect
    StyleInter(pivot_table, 'group-volatility', fig_id='S2A')     # Fig S2A
    StyleInter(pivot_table, 'group-feedback', fig_id='S2B')       # Fig S2B
    StyleInter(pivot_table, 'volatility-feedback', fig_id='S2C')  # Fig S2C
    
    # Fig S3: Understand the flexible behaviors
    HumanAda('gain', fig_id='S3')   # Fig S3

    show_bms()