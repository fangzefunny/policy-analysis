import os 
import numpy as np 
import pandas as pd 
from scipy.stats import ttest_ind, pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols 
from statsmodels.stats.anova import anova_lm

from tabulate import tabulate
import pingouin as pg 

import seaborn as sns 
import matplotlib.pyplot as plt 

from utils.model import *
from utils.viz import viz 


# path to the current file 
path = os.path.dirname(os.path.abspath(__file__))

def model_fit(models, method='mle'):
    feedbacks = ['gain', 'loss']
    crs = {}
    for m in models:
        for feedback in feedbacks:
            fname = f'{path}/../simulations/exp1data/{m}/sim-{feedback}_exp1data-{method}-idx0.csv'
            data  = pd.read_csv(fname)
            n_param = eval(m).n_params
            subj_Lst = data['sub_id'].unique()

            nlls, aics = [], [] 
            for sub_id in subj_Lst:
                sel_data = data.query(f'sub_id=="{sub_id}" & feedback_type=="{feedback}"')
                inll = sel_data['logLike'].sum() 
                nlls.append(inll)
                if np.isnan(inll): print(inll) 
                aics.append(2*inll + 2*n_param)

            crs[m] = {'nll': nlls, 'aic': aics}
    return crs

def model_cmp(quant_crs):
    crs = ['nll', 'aic']
    pairs = [['gagModel', 'risk'],
             ['gagModel', 'mix_pol_3w'],
             ['risk',  'mix_pol_3w']]
    
    for cr in crs:
        print(f'''
            ------------- {cr} ------------- ''')
        for p in pairs:
            x = quant_crs[p[0]][cr]
            y = quant_crs[p[1]][cr]
            res = ttest_ind(x, y)
            print(f'''
            {p[0]}-{p[1]}: 
                {p[0]}:{np.mean(x):.3f}, {p[1]}:{np.mean(y):.3f}
                t={res[0]:.3f} p={res[1]:.3f}''')

def t_test(x_data, y_data, paired=False, title=''):
    df = pg.ttest(x_data, y_data, paired=paired)
    dof = df.loc[:, 'dof'].values[0]
    t   = df.loc[:, 'T'].values[0]
    pval = df.loc[:, 'p-val'].values[0]
    cohen_d = df.loc[:, 'cohen-d'].values[0]
    pair_str = '-paired' if paired==True else ''
    print(f'{title} \tt{pair_str}({dof:.3f})={t:.3f}, p={pval:.3f}, cohen-d={cohen_d:.3f}')

def corr(x_data, y_data, title=''):
    df = pg.corr(x_data, y_data)
    n = df.loc[:, 'n'].values[0]
    r   = df.loc[:, 'r'].values[0]
    pval = df.loc[:, 'p-val'].values[0]
    print(f'{title} \tr({n})={r:.3f}, p={pval:.3f}')

def anova(dv, between, data, all_table=False):
    df = pg.anova(dv=dv, between=between, data=data).rename(columns={'p-unc': 'punc'})
    if all_table:
       print(df.round(3).to_string())
    else:
        sig_df = df.query('punc<=.05')
        dof2 = int(df.query('Source=="Residual"')['DF'].values[0])
        other_min_p = df.query('punc>.05')['punc'].min()
        for _, row in sig_df.iterrows():
            title = row['Source']
            dof   = int(row['DF'])
            F     = row['F']
            p     = row['punc']
            np2   = row['np2']
            print(f'\t{title}:\tF({dof}, {dof2})={F:.3f}, p={p:.3f}, np2={np2:.3f}')
        print(f'\tOther: \tp>={other_min_p:.3f}')

def main_effect(pivot_table, pred, cond1, cond2,
            tar=['l1', 'l2', 'l3', 'l4'], 
            notes=['exp utility', 'reward probability', 'magnitude', 'habit']):
    nr, nc = 1, len(tar)
    fig, axs = plt.subplots(nr, nc, figsize=(nc*3.7, nr*4), sharey=True, sharex=True)
    for_title = t_test(pivot_table, cond1, cond2, tar=tar)
    for idx in range(nc):
        ax  = axs[idx]
        sns.boxplot(x=pred, y=f'{tar[idx]}', data=pivot_table,
                        palette=viz.Palette, ax=ax)
        ax.set_xlim([-.8, 1.8])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(f'{notes[idx]} {for_title[idx]}')
        # if idx == 1: ax.legend(bbox_to_anchor=(1.4, 0), loc='lower right')
        # else: ax.get_legend().remove()
    plt.tight_layout()
    plt.show()

def f_twoway(data, fac1, fac2, tar=['l1', 'l2', 'l3']):
    ## the significant test 
    for_title = []
    for i in tar:
        model = ols(f'{i} ~ C({fac1}) + C({fac2}) + C({fac1}):C({fac2})', data).fit()
        res = anova_lm(model).loc[f'C({fac1}):C({fac2})', ['F', 'PR(>F)']]
        if res[1] < .01:
            for_title.append('**')
        elif res[1] < .05:
            for_title.append('*')
        else:
            for_title.append('')
        print(f'{i} f-two way: f={res[0]:.4f}, p-val:{res[1]:.4f}')
    return for_title

def intersect_effect(pivot_table, fac1, fac2,
            tar=['l1', 'l2', 'l3', 'l4'], 
            notes=['exp utility', 'reward probability', 'magnitude', 'habit']):
    nr, nc = 1, len(tar)
    fig, axs = plt.subplots(nr, nc, figsize=(nc*3.7, nr*4), sharey=True)
    for_title = f_twoway(pivot_table, fac1, fac2, tar)
    for idx in range(nc):
        ax  = axs[idx]
        sns.boxplot(x=fac1, y=f'{tar[idx]}', data=pivot_table,
                        hue=fac2, palette=viz.Palette, ax=ax)
        ax.set_xlim([-.8, 1.8])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(f'{notes[idx]} {for_title[idx]}')
        if idx == nc-1: ax.legend(bbox_to_anchor=(1.6, .5), loc='right')
        else: ax.get_legend().remove()
    plt.tight_layout()
    plt.show()

def pred_syndrome(pivot_table, pred='ratioanl_deg'):
    nr, nc = 1, 3
    syns = ['g', 'f1', 'f2']
    fix, axs = plt.subplots(nr, nc, figsize=(nc*3.4, nr*4), sharey=True)
    for i, syn in enumerate(syns):
        ax = axs[i]
        sns.scatterplot(x=pred, y=syn, data=pivot_table, ax=ax)
        res = pearsonr(pivot_table[pred], pivot_table[syn])
        if res[1] < .05:
            x = sm.add_constant(pivot_table[pred])
            params = sm.OLS(x, pivot_table[syn]).fit().params
            x = pivot_table[pred].values
            y = params.iloc[0, 0] + x*params.iloc[0, 1] 
            print(f'{syn} ~ {params.iloc[0, 1]:.4f}*{pred} + {params.iloc[0, 0]:.4f}')
            sns.lineplot(x=x, y=y, color='k', ax=ax)
        print(f'{syn}: r={res[0]:.4f}, pval={res[1]:.4f}')
        ax.set_title(f'{syn}')
        ax.set_ylabel('')
    plt.tight_layout()


    # def build_pivot_table(method, agent='MOS', min_q=0, max_q=1, verbose=True):
#     features = eval(agent).voi
#     exp1data = pd.read_csv(f'{path}/../simulations/exp1data/{agent}/sim-{method}.csv')
#     sub_syndrome = pd.read_csv(f'{path}/../data/bifactor.csv')
#     sub_syndrome = sub_syndrome.rename(columns={'Unnamed: 0': 'sub_id', 'F1.': 'f1', 'F2.':'f2'})
#     gby = ['sub_id', 'trial_type', 'feedback_type', 'group']
#     pivot_table  = exp1data.groupby(by=gby)[features].mean().reset_index()
    
#     #datainfo(pivot_tables)
#     if verbose: print('#-------- Clean Outliers ---------- #\n')
#     # concate to build a table
#     pivot_table['log_alpha'] = pivot_table['alpha'].apply(lambda x: np.log(1/(1+np.exp(-x))))
#     oldN = pivot_table.shape[0]

#     # remove the outliers
#     tar = ['log_alpha'] + features[1:]
#     for i in tar:
#         qhigh = pivot_table[i].quantile(max_q)
#         qlow  = pivot_table[i].quantile(min_q)
#         pivot_table = pivot_table.query(f'{i}<{qhigh} & {i}>{qlow}')
#     if verbose:
#         print(f'    {pivot_table.shape[0]} rows')
#         print(f'    {oldN - pivot_table.shape[0]} rows have been deleted')
#         print(f'    {pivot_table.shape[0] * 100/ oldN:.1f}% data has been retained')

#     # add syndrome 
#     pivot_table = pivot_table.join(sub_syndrome.set_index('sub_id'), 
#                         on='sub_id', how='left')
#     for i in ['g', 'f1', 'f2']:
#         pivot_table[i] = pivot_table[i].fillna(pivot_table[i].mean())

#     return pivot_table