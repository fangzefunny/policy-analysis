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
viz.default_img_set()

# set up path 
path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f'{path}/figures'): os.mkdir(f'{path}/figures')

feedback_types = ['gain', 'loss']
patient_groups = ['HC', 'PAT']
block_types    = ['sta', 'vol']
policies       = ['EU', 'MO', 'HA']
dpi = 300
width = .7

# ------------  WRITE STATISTICS  ------------ #

def write_stats():

    with open(f'{path}/figures/statistics.txt', 'w')as f:

        ## The MOS6 model 
        data = build_pivot_table('bms', agent='MOS_fix', min_q=.01, max_q=.99)
        data['group'] = data['group'].map(
                        {'HC': 'HC', 'MDD': 'PAT', 'GAD': 'PAT'})

        ## Check lambda HC
        tar = ['l1']
        sel_data = data.groupby(by=['sub_id', 'group'])[tar].mean().reset_index()
        f.write('\n'
               +'################################################################\n'
               +'#                      MOS6-λ_EU:  HCxPAT                      #\n'
               +'################################################################\n\n')
        f.writelines(t_test(sel_data, f'group=="HC"', f'group=="PAT"', tar=tar))

        ## Check lambda MO
        tar = ['l2']
        sel_data = data.groupby(by=['sub_id', 'group'])[tar].mean().reset_index()
        f.write('\n\n\n\n'
               +'################################################################\n'
               +'#                      MOS6-λ_MO:  HCxPAT                      #\n'
               +'################################################################\n\n')
        f.writelines(t_test(sel_data, f'group=="HC"', f'group=="PAT"', tar=tar))

        ## Check lambda HC
        tar = ['l3']
        sel_data = data.groupby(by=['sub_id', 'group'])[tar].mean().reset_index()
        f.write('\n\n\n\n'
               +'################################################################\n'
               +'#                      MOS6-λ_HA:  HCxPAT                      #\n'
               +'################################################################\n\n')
        f.writelines(t_test(sel_data, f'group=="HC"', f'group=="PAT"', tar=tar))

        ## Check learning rate 
        sel_data = data.groupby(by=['sub_id', 'group'])['log_alpha'].mean().reset_index()
        f.write('\n\n\n\n'
               +'################################################################\n'
               +'#                     MOS6-log α:  HCxPAT                      #\n'
               +'################################################################\n\n')
        f.writelines(t_test(sel_data, f'group=="HC"', f'group=="PAT"', tar=['log_alpha']))

        ## The MOS18 model 
        data = build_pivot_table('bms', agent='MOS', min_q=.01, max_q=.99)
        data['group'] = data['group'].map(
                        {'HC': 'HC', 'MDD': 'PAT', 'GAD': 'PAT'})

        ## learning rate: 2x2x2 anova
        tar = 'log_alpha'
        sel_data = data.groupby(by=['sub_id', 'group', 'b_type', 'feedback_type']
                )[tar].mean().reset_index()
        f.write('\n\n\n\n'
               +'################################################################\n'
               +'#                     MOS18-log α:  ANOVA                      #\n'
               +'################################################################\n\n')
        f.writelines([pg.anova(dv=tar, 
            between=['group', 'b_type', 'feedback_type'], data=data).round(3).to_string()])
        
        ## λ1: 2x2x2 anova
        tar = 'l1'
        sel_data = data.groupby(by=['sub_id', 'group', 'b_type', 'feedback_type']
                )[tar].mean().reset_index()
        f.write('\n\n\n\n'
               +'################################################################\n'
               +'#                      MOS18-λ_HC:  ANOVA                      #\n'
               +'################################################################\n\n')
        f.writelines([pg.anova(dv=tar, 
            between=['group', 'b_type', 'feedback_type'], data=data).round(3).to_string()])

        ## λ2: 2x2x2 anova
        tar = 'l2'
        sel_data = data.groupby(by=['sub_id', 'group', 'b_type', 'feedback_type']
                )[tar].mean().reset_index()
        f.write('\n\n\n\n'
               +'################################################################\n'
               +'#                      MOS18-λ_MO:  ANOVA                      #\n'
               +'################################################################\n\n')
        f.writelines([pg.anova(dv=tar, 
            between=['group', 'b_type', 'feedback_type'], data=data).round(3).to_string()])

        ## λ3: 2x2x2 anova
        tar = 'l3'
        sel_data = data.groupby(by=['sub_id', 'group', 'b_type', 'feedback_type']
                )[tar].mean().reset_index()
        f.write('\n\n\n\n'
               +'################################################################\n'
               +'#                      MOS18-λ_HA:  ANOVA                      #\n'
               +'################################################################\n\n')
        f.writelines([pg.anova(dv=tar, 
            between=['group', 'b_type', 'feedback_type'], data=data).round(3).to_string()])


# ------------  MODEL COMPARISON ------------- #

def ModelComp(data_set, models, ticks, fig_id):
    
    ## get BMS
    fit_sub_info = []
    for i, m in enumerate(models):
        with open(f'fits/{data_set}/fit_sub_info-{m}-bms.pkl', 'rb')as handle:
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

     ## get nll, aic ,bic 
    cols = ['NLL', 'AIC', 'BIC', 'model', 'sub_id']
    crs = {k: [] for k in cols}
    for m in models:
        with open(f'fits/{data_set}/fit_sub_info-{m}-bms.pkl', 'rb')as handle:
            fit_info = pickle.load(handle)
        # get the subject list 
        if i==0: subj_lst = fit_info.keys() 
        # get log post
        nll = [-fit_info[idx]['log_like'] for idx in subj_lst]
        aic = [fit_info[idx]['aic'] for idx in subj_lst]
        bic = [fit_info[idx]['bic'] for idx in subj_lst]
        crs['NLL'] += nll
        crs['AIC'] += aic
        crs['BIC'] += bic
        crs['model'] += [m]*len(nll)
        crs['sub_id'] += list(subj_lst)
    #crs['PXP'] = bms_results['pxp']
    crs = pd.DataFrame.from_dict(crs)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    xx = list(range(len(models)))
    for i, c in enumerate(['NLL', 'AIC', 'BIC']):
        ax = axs[i//2, i%2]
        best_model = crs.groupby(by='model')[c].mean().reset_index(
                        ).sort_values(c)['model'].values[0]
        best_val = len(models)*list(crs.query(f'model=="{best_model}"')[c].values)
        crs[f'delta {c}'] = crs[c] - best_val
        sns.barplot(x='model', y=f'delta {c}', data=crs, capsize=.2, errwidth=2,
                    palette=viz.divPalette[:len(models)], ax=ax)
        ax.set_xticks(xx)
        ax.set_xticklabels(ticks, rotation=0, fontsize=11)
        ax.set_xlim([0-.8, len(models)-1+.8])
        ax.set_ylabel('\n'+r'$\Delta$'+f'{c}')
        #ax.set_ylabel('\n'+r'$\Delta$'+f'{c}') if i < 3 else ax.set_ylabel(f'\n{c}')
        ax.set_xlabel(' ')

    ax = axs[1, 1]
    sns.barplot(x=xx, y=bms_results['pxp'], 
                palette=viz.divPalette[:len(models)], ax=ax)
    ax.set_xticks(xx)
    ax.set_xticklabels(ticks, rotation=0, fontsize=15)
    ax.set_xlim([0-.8, len(models)-1+.8])
    ax.set_ylabel('PXP')
    ax.set_xlabel(' ')
    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig{fig_id}_quant_{data_set}.pdf', dpi=300)

# ---------- MODEL-BASE ANALYSIS ----------- #

def StylexConds(data, cond, fig_id, mode='fix'):
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

    # prepare the inputs 
    tars = ['l1', 'l2', 'l3']
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    gby = ['sub_id', varr] if mode=='fix' else [
        'sub_id', 'group', 'feedback_type', 'b_type']
    data = data.groupby(by=gby)[tars].mean().reset_index()
    yticks = [r'$\lambda_{EU}$',r'$\lambda_{MO}$',r'$\lambda_{HA}$']

    # show bar plot 
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True, sharex=True)
    t_test(data, f'{varr}=="{case1}"', f'{varr}=="{case2}"', tar=tars)

    for i, t in enumerate(tars):
        ax = axs[i]
        sns.boxplot(x=varr, y=t, data=data, 
                width=width, order=[case1, case2],
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
                        palette=colors, alpha=0.5, order=[case1, case2],
                        ax=ax)
        ax.set_ylim([-5, 5])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(ticks)
        ax.set_xlabel('group')
        ax.set_ylabel(f'{yticks[i]}')
        ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig{fig_id}_Stylex{cond}.pdf', dpi=300)

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
    plt.savefig(f'{path}/figures/Fig{fig_id}_Stylex{cond}.pdf', dpi=300)

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
    predlabels = [r'$\lambda_{EU}$',r'$\lambda_{MO}$',r'$\lambda_{HA}$']
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data = data.groupby(by=['sub_id'])[['g', 'l1', 'l2', 'l3']].mean().reset_index()

    nr, nc = 1, int(len(tars))
    fig, axs = plt.subplots(nr, nc, figsize=(10, 3), sharey=True)
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
        ax.set_ylabel(f'General factor (a.u.)')
        ax.set_xlabel(predlabels[i])
        ax.set_box_aspect(1)
 
    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig{fig_id}_PrefxSyndrome.pdf', dpi=300)

def LRxConds(data, cond, fig_id, mode='fix'):
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
    gby = ['sub_id', varr] if mode=='fix' else [
        'sub_id', 'group', 'feedback_type', 'b_type']
        
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

    # check t test 
    data = data.groupby(by=gby)[tars].mean().reset_index()
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
    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig{fig_id}_LRx{cond}.pdf', dpi=dpi)

def StrategyAda(fig_id):

    fname = f'{path}/simulations/exp1data/MOS_fix/simsubj-exp1data-sta_first-AVG.csv'
    data = pd.read_csv(fname)

    data = data.groupby(by=['trials'])[['l1_effect', 'l2_effect', 'l3_effect']].mean()
    psi  = np.zeros([180])
    psi[:90]     = .7
    psi[90:110]  = .2
    psi[110:130] = .8
    psi[130:150] = .2
    psi[150:170] = .8
    psi[170:180] = .2

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3))
    ax = ax 
    labels = ['EU', 'MO', 'HA']
    for i in range(3):
        sns.lineplot(x='trials', y=f'l{int(i)+1}_effect',
                    data=data, color=viz.Palette2[i], label=labels[i], ax=ax)
    sns.lineplot(x=np.arange(180), y=psi, color='k', ls='--', ax=ax)
    ax.set_ylabel('Prob. of choosing \nthe left stimulus')
    ax.set_xlabel('Trials')
    ax.set_ylim([-.1, 1.1])
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig{fig_id}_StrategySim.pdf', dpi=dpi)

def PolicyAda(fig_id):

    psi  = np.zeros([180])
    psi[:90]     = .7
    psi[90:110]  = .2
    psi[110:130] = .8
    psi[130:150] = .2
    psi[150:170] = .8
    psi[170:180] = .2

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3))
    ax = ax 
    for i, g in enumerate(['HC', 'PAT']):
        fname = f'{path}/simulations/exp1data/MOS_fix/simsubj-exp1data-sta_first-{g}.csv'
        data = pd.read_csv(fname)
        data = data.groupby(by=['trials'])[['pi']].mean()
        sns.lineplot(x='trials', y=f'pi', data=data, 
                    color=viz.PurplePairs[i], label=g)
    sns.lineplot(x=np.arange(180), y=psi, color='k', ls='--')
    ax.set_ylabel('Prob. of choosing \nthe left stimulus')
    ax.set_xlabel('Trials')
    ax.legend()
    ax.set_ylim([-.1, 1.1])
    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig{fig_id}_PolicySim.pdf', dpi=dpi)

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
             
    cs = ['group=="HC"', 'group!="HC"']
    sel_data = pd.concat(cases['sta0.7-vol0.2']).reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3))
    sns.lineplot(x=np.arange(180), y=psi, color='k', ls='--', ax=ax)
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
        sns.lineplot(x='trial', y='humanAct', data=sdata, color=viz.PurplePairs[i], ci=0, label=lbs[i])
    ax.legend()
    ax.set_ylim([-.1, 1.1])
    ax.set_xlabel('Trials')
    ax.set_ylabel('Prob. of choosing \nthe left stimulus')
    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig{fig_id}_HumanSim-{mode}.pdf', dpi=dpi)

# ----------- PARAMETER RECOVERY ----------- #

def plot_param_recovery(model, fig_id):

    ## load ground truth parameters
    fname = f'{path}/data/params_truth-exp1data-{model}.csv'
    truth = pd.read_csv(fname, index_col=False)
    truth = truth.rename(columns={
        'α': 'alpha_tr', 'λ1': 'l1_tr', 'λ2': 'l2_tr', 'λ3': 'l3_tr'})
    truth['alpha_tr'] = truth['alpha_tr']

    ## load recovered parameters 
    fname =  f'{path}/fits/param_recovery-MOS_fix/'
    fname += f'fit_sub_info-{model}-bms.pkl'
    with open(fname, 'rb')as handle: r_info = pickle.load(handle)
    param_name = r_info[list(r_info.keys())[0]]['param_name']
    recovered  = {k: [] for k in param_name}
    recovered['sub_id'] = []
    for sub_id in r_info.keys():
        recovered['sub_id'].append(sub_id)
        for i, pname in enumerate(param_name):
            recovered[pname].append(r_info[sub_id]['param'][i])
    recovered = pd.DataFrame(recovered)
    recovered = recovered.rename(columns={
        'α': 'alpha_re', 'λ1': 'l1_re', 'λ2': 'l2_re', 'λ3': 'l3_re'})
    recovered['alpha_re'] = recovered['alpha_re']

    # combine data set 
    data = truth.merge(recovered, on='sub_id')

    # plot: vary learning rate condition
    data_types = ['vary_lr', 'vary_w']
    fig_ind = ['A', 'B']
    colors = [viz.BluePairs]+[viz.PurplePairs]*3
   
    fig, axs = plt.subplots(2, 4, figsize=(15, 7))

    for j, data_type in enumerate(data_types):
        
        if data_type == 'vary_lr': 
            xvarrs = ['alpha_tr', 'l1_re', 'l2_re', 'l3_re']
            xticks = ['Ground truth '+r'$\alpha$', 'Recovered '+r'$\lambda_{EU}$', 
                        'Recovered '+r'$\lambda_{MO}$', 'Recovered '+r'$\lambda_{HA}$']
            yvarrs = ['alpha_re']*4
            yticks = ['Recovered '+r'$\alpha$']*4
        elif data_type == 'vary_w': 
            xvarrs = ['l2_re', 'l1_tr', 'l2_tr', 'l3_tr']
            xticks = ['Recovered '+r'$\lambda_{MO}$', 'Ground truth '+r'$\lambda_{EU}$', 
                        'Ground truth '+r'$\lambda_{MO}$', 'Ground truth '+r'$\lambda_{HA}$']
            yvarrs = ['alpha_re', 'l1_re', 'l2_re', 'l3_re']
            yticks = ['Recovered '+r'$\alpha$', 'Recovered '+r'$\lambda_{EU}$', 
                        'Recovered '+r'$\lambda_{MO}$', 'Recovered '+r'$\lambda_{HA}$']

        for i in range(len(xvarrs)):
            ax = axs[j, i]
            # check correlation 
            x = data.query(f'data_type=="{data_type}"')[f'{xvarrs[i]}']
            y = data.query(f'data_type=="{data_type}"')[f'{yvarrs[i]}']
            print(f'{xvarrs[i]} x {yvarrs[i]}:')
            print(pg.corr(x.values, y.values).round(3))
            sel_data = data.query(f'data_type=="{data_type}"').groupby(
                        by=[f'{xvarrs[i]}', f'{yvarrs[i]}']).mean(
                            numeric_only=True).reset_index()
            if j==0: sel_data = sel_data.sort_values(by='alpha_tr')
            sns.scatterplot(x=f'{xvarrs[i]}', 
                            y=f'{yvarrs[i]}',
                            data=sel_data,
                            color=colors[i][0], s=60, ax=ax)
            ax.plot(sel_data[f'{xvarrs[i][:-2]+"tr"}'].values, 
                    sel_data[f'{yvarrs[i][:-2]+"tr"}'].values,
                    color='k', ls='--', lw=.5)
            ax.set_xlabel(f'{xticks[i]}')
            ax.set_ylabel(f'{yticks[i]}')
            #ax.set_title(xticks[i])
            ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig{fig_id}_param_recovery-{data_type}.pdf', dpi=dpi)

# ----------- MODEL RECOVERY ----------- #

def plot_model_recovery(data_set, models, ticks, fig_id):

    ## get nll, aic ,bic 
    cols = ['NLL', 'AIC', 'BIC', 'model', 'sub_id']
    crs = {k: [] for k in cols}
    for i, m in enumerate(models):
        with open(f'fits/{data_set}/fit_sub_info-{m}-bms.pkl', 'rb')as handle:
            fit_info = pickle.load(handle)
        # get the subject list 
        if i==0: subj_lst = fit_info.keys() 
        # get log post
        nll = [-fit_info[idx]['log_like'] for idx in subj_lst]
        aic = [fit_info[idx]['aic'] for idx in subj_lst]
        bic = [fit_info[idx]['bic'] for idx in subj_lst]
        crs['NLL'] += nll
        crs['AIC'] += aic
        crs['BIC'] += bic
        crs['model'] += [m]*len(nll)
        crs['sub_id'] += list(subj_lst)
    crs = pd.DataFrame.from_dict(crs)

    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.5))
    xx = list(range(len(models)))
    for i, c in enumerate(['AIC', 'BIC']):
        ax = axs[i]
        best_model = crs.groupby(by='model')[c].mean().reset_index(
                        ).sort_values(c)['model'].values[0]
        best_val = len(models)*list(crs.query(f'model=="{best_model}"')[c].values)
        crs[f'delta {c}'] = crs[c] - best_val
        sns.barplot(x='model', y=f'delta {c}', data=crs, capsize=.2, errwidth=2,
                    palette=viz.divPalette[:len(models)], ax=ax)
        ax.set_xticks(xx)
        ax.set_xticklabels(ticks, rotation=0, fontsize=10)
        ax.set_xlim([0-.8, len(models)-1+.8])
        ax.set_ylabel('\n'+r'$\Delta$'+f'{c}')
        #ax.set_ylabel('\n'+r'$\Delta$'+f'{c}') if i < 3 else ax.set_ylabel(f'\n{c}')
        ax.set_xlabel(' ')
    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig{fig_id}_model_recovery_{data_set}.pdf', dpi=300)
  

if __name__ == '__main__':

    # --------- Data stats  --------- #

    write_stats()

    # --------- Main results --------- #

    pivot_table = build_pivot_table('bms', agent='MOS_fix', min_q=.01, max_q=.99)
    pivot_table['group'] = pivot_table['group'].map(
                    {'HC': 'HC', 'MDD': 'PAT', 'GAD': 'PAT'})

    # Fig 2: quantitative fit table 
    ModelComp('exp1data', models=['MOS_fix', 'FLR_fix', 'RP_fix', 'MOS', 'FLR', 'RP'],
               ticks=['MOS6', 'FLR6', 'RS6', 'MOS18', 'FLR15', 'RS9'], fig_id='2') 
    plt.close('all')
    
    # Fig 3: Decision style effect
    StylexConds(pivot_table, 'group', fig_id='3A')   # Fig 3A
    StylexSyndrome(pivot_table, fig_id='3B')         # Fig 3B
    plt.close('all')

    # Fig 4: Understand the flexible behaviors
    HumanAda('loss', fig_id='4A')                    # Fig 4A
    PolicyAda(fig_id='4B')                           # Fig 4B
    StrategyAda(fig_id='4C')                         # Fig 4C
    plt.close('all')

    # Fig 5: param recovery
    plot_param_recovery(model='MOS_fix', fig_id='5') 
    plt.close('all')
    
    # Fig 6: model recovery 
    plot_model_recovery('exp1data-MOS_fix', 
                models=['MOS_fix', 'FLR_fix', 'RP_fix', 'MOS', 'FLR', 'RP'],
                ticks=['MOS6', 'FLR6', 'RS6', 'MOS18', 'FLR15', 'RS9'], fig_id='6')
    plt.close('all')

    # # ------ Supplementary materials ------- #

    # Fig S1: learning rate effect 
    pivot_table = build_pivot_table('bms', agent='MOS', min_q=.01, max_q=.99)
    pivot_table['group'] = pivot_table['group'].map(
                    {'HC': 'HC', 'MDD': 'PAT', 'GAD': 'PAT'})
    StylexConds(pivot_table, 'group', fig_id='S1A', mode='vary')   # Fig 3A
    StylexSyndrome(pivot_table, fig_id='S1B')         # Fig 3B
    plt.close('all')

    # Fig S3: Understand the flexible behaviors
    HumanAda('gain', fig_id='S2')   # Fig S3
    plt.close('all')

