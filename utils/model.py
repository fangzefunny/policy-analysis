import numpy as np 
import pandas as pd 
 
from scipy.special import softmax 
from scipy.stats import gamma, uniform, beta

from utils.fit import *
from utils.env_fn import rl_reversal
from utils.viz import *

eps_ = 1e-13
max_ = 1e+13

# ------------------------------#
#        Axulliary funcs        #
# ------------------------------#

flatten = lambda l: [item for sublist in l for item in sublist]

def get_param_name(params, block_types=['sta', 'vol'], feedback_types=['gain', 'loss']):
    return flatten([flatten([[f'{key}_{j}_{i}' for key in params]
                                               for i in feedback_types])
                                               for j in block_types])

def clip_exp(x):
    x = np.clip(x, a_min=-max_, a_max=50)
    return np.exp(x) 

sigmoid = lambda x: 1 / (1+clip_exp(-x))

# ------------------------------#
#         Agent wrapper         #
# ------------------------------#

class wrapper:
    '''Agent wrapper

    We use the wrapper to

        * Fit
        * Simulate
        * Evaluate the fit 
    '''

    def __init__(self, agent, env_fn):
        self.agent  = agent
        self.env_fn = env_fn
        self.use_hook = False
    
    # ------------ fit ------------ #

    def fit(self, data, method, alg, pool=None, p_priors=None,
            init=False, seed=2021, verbose=False, n_fits=40):
        '''Fit the parameter using optimization 
        '''

        # get functional inputs 
        fn_inputs = [self.loss_fn, 
                     data, 
                     self.agent.p_bnds,
                     self.agent.p_pbnds, 
                     self.agent.p_name,
                     self.agent.p_priors if p_priors is None else p_priors,
                     method,
                     alg, 
                     init,
                     seed,
                     verbose]
        
        if pool:
            sub_fit = fit_parallel(pool, *fn_inputs, n_fits=n_fits)
        else: 
            sub_fit = fit(*fn_inputs)  

        return sub_fit      

    def loss_fn(self, params, sub_data, p_priors=None):
        '''Total likelihood

        Fit individual:
            Maximum likelihood:
            log p(D|θ) = log \prod_i p(D_i|θ)
                       = \sum_i log p(D_i|θ )
            or Maximum a posterior 
            log p(θ|D) = \sum_i log p(D_i|θ ) + log p(θ)
        '''
        # negative log likelihood
        tot_loglike_loss  = -np.sum([self.loglike(params, sub_data[key])
                    for key in sub_data.keys()])
        # negative log prior 
        tot_logprior_loss = 0 if p_priors==None else \
            -self.logprior(params, p_priors)
        # sum
        return tot_loglike_loss + tot_logprior_loss

    def loglike(self, params, block_data):
        '''Likelihood for one sample
        -log p(D_i|θ )
        In RL, each sample is a block of experiment,
        Because it is independent across experiment.
        '''
        # init subject and load block type
        block_type = block_data.loc[0, 'block_type']
        env  = self.env_fn(block_type)
        subj = self.agent(env, params)
        ll   = 0
       
        ## loop to simulate the responses in the block 
        for _, row in block_data.iterrows():

            # predict stage: obtain input
            ll += env.eval_fn(row, subj)

        return ll
          
    def logprior(self, params, p_priors):
        '''Add the prior of the parameters
        '''
        lpr = 0
        for pri, param in zip(p_priors, params):
            lpr += np.max([pri.logpdf(param), -max_])
        return lpr

    # ------------ evaluate ------------ #

    def eval(self, data, params):
        sim_data = [] 
        for block_id in data.keys():
            block_data = data[block_id].copy()
            sim_data.append(self.eval_block(block_data, params))
        return pd.concat(sim_data, ignore_index=True)
    
    def eval_block(self, block_data, params):

        # init subject and load block type
        block_type = block_data.loc[0, 'block_type']
        env  = self.env_fn(block_type)
        subj = self.agent(env, params)

        ## init a blank dataframe to store variable of interest
        col = ['ll'] + self.agent.voi
        init_mat = np.zeros([block_data.shape[0], len(col)]) + np.nan
        pred_data = pd.DataFrame(init_mat, columns=col)  

        ## loop to simulate the responses in the block
        for t, row in block_data.iterrows():

            # record some insights of the model
            # for v in self.agent.voi:
            #     pred_data.loc[t, v] = eval(f'subj.get_{v}()')

            # simulate the data 
            ll = env.eval_fn(row, subj)
            
            # record the stimulated data
            pred_data.loc[t, 'll'] = ll

        # drop nan columns
        pred_data = pred_data.dropna(axis=1, how='all')
            
        return pd.concat([block_data, pred_data], axis=1)

    # ------------ simulate ------------ #

    def sim(self, data, params, rng):
        sim_data = [] 
        for block_id in data.keys():
            block_data = data[block_id].copy()
            for v in self.env_fn.voi:
                if v in block_data.columns:
                    block_data = block_data.drop(columns=v)
            sim_data.append(self.sim_block(block_data, params, rng))
        
        return pd.concat(sim_data, ignore_index=True)

    def sim_block(self, block_data, params, rng):

        # init subject and load block type
        block_type = block_data.loc[0, 'block_type']
        env  = self.env_fn(block_type)
        subj = self.agent(env, params)

        ## init a blank dataframe to store variable of interest
        col = self.env_fn.voi + self.agent.voi
        init_mat = np.zeros([block_data.shape[0], len(col)]) + np.nan
        pred_data = pd.DataFrame(init_mat, columns=col)  

        ## loop to simulate the responses in the block
        for t, row in block_data.iterrows():

            # simulate the data 
            subj_voi = env.sim_fn(row, subj, rng)

            # record some insights of the model
            for i, v in enumerate(self.agent.voi):
                pred_data.loc[t, v] = eval(f'subj.get_{v}()')

            # if register hook to get the model insights
            if self.use_hook:
                for k in self.insights.keys():
                    self.insights[k].append(eval(f'subj.get_{k}()'))

            # record the stimulated data
            for i, v in enumerate(env.voi): 
                pred_data.loc[t, v] = subj_voi[i]

        # drop nan columns
        pred_data = pred_data.dropna(axis=1, how='all')
            
        return pd.concat([block_data, pred_data], axis=1)
    
    def register_hooks(self, *args):
        self.use_hook = True 
        self.insights = {k: [] for k in args}

# ------------------------------#
#         Memory buffer         #
# ------------------------------#

class simpleBuffer:
    '''Simple Buffer 2.0
    Update log: 
        To prevent naive writing mistakes,
        we turn the list storage into dict.
    '''
    def __init__(self):
        self.m = {}
        
    def push(self, m_dict):
        self.m = {k: m_dict[k] for k in m_dict.keys()}
        
    def sample(self, *args):
        lst = [self.m[k] for k in args]
        if len(lst)==1: return lst[0]
        else: return lst

# ------------------------------#
#          Base model           #
# ------------------------------#

class baseAgent:
    '''Base Agent'''
    name     = 'base'
    n_params = 0
    p_bnds   = None
    p_pbnds  = []
    p_name   = []  
    n_params = 0 
    p_priors = None 
    # value of interest, used for output
    # the interesting variable in simulation
    voi      = []
    
    def __init__(self, nA, params):
        self.nA = nA 
        self.load_params(params)
        self._init_believes()
        self._init_buffer()

    def load_params(self, params): 
        return NotImplementedError

    def _init_buffer(self):
        self.mem = simpleBuffer()
    
    def _init_believes(self):
        self._init_critic()
        self._init_actor()
        self._init_dists()

    def _init_critic(self): pass 

    def _init_actor(self): pass 

    def _init_dists(self):  pass

    def learn(self): 
        return NotImplementedError

    def policy(self, m, **kwargs):
        '''Control problem
            create a policy that map to
            a distribution of action 
        '''
        return NotImplementedError

class RL(baseAgent):
    name     = 'RL'
    p_bnds   = None
    p_pbnds  = [(-2, 2)] + [(-10, -.15), (-10, -.15)]*4
    p_name   = ['β'] + get_param_name(['α+', 'α_'])
    p_priors = []
    p_trans  = [lambda x: clip_exp(x), 
                lambda x: 1/(1+clip_exp(-x)), lambda x: 1/(1+clip_exp(-x))]*4
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence']
    color    = viz.r2 
   
    def load_params(self, params):
        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]
        # assign the parameter
        self.beta           = params[0]
        self.a_pos_sta_gain = params[1]
        self.a_neg_sta_loss = params[2]
        self.a_pos_vol_gain = params[3]
        self.a_neg_vol_loss = params[4]
        
    def _init_critic(self):
        self.p1     = 1/2
        self.p_S   = np.array([1-self.p1, self.p1]) 

    def learn(self):
        self._learn_critic()

    def _learn_critic(self):
        s, t, f = self.mem.sample('s', 't_type', 'f_type')
        delta = s-self.p1
        o = 'pos' if delta>0 else 'neg'
        self.o = o 
        self.p1 += eval(f'self.a_{o}_{t}_{f}') * delta 
        self.p_S = np.array([1-self.p1, self.p1])

    def policy(self, m, **kwargs):
        self.pi = softmax(self.beta*self.p_S*m)
        return self.pi

    def get_pS1(self):
        return self.p_S[1]
    
    def get_pi1(self):
        return self.pi[1]
    
    def get_alpha(self):
        t, f = self.mem.sample('t_type', 'f_type')
        return eval(f'self.a_{self.o}_{t}_{f}')

    def get_valence(self):
        f_type = self.mem.sample('f_type')
        if f_type=="gain":
            if self.o=='pos':
                return 'good outcome'
            else:
                return 'bad outcome'
        elif f_type=="loss":
            if self.o=='pos':
                return 'bad outcome'
            else:
                return 'good outcome'

# ------------------------------------------ #
#       Flexible learning rate model         #
# ------------------------------------------ #
    
class FLR19(RL):
    name     = 'FLR19'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2), (-2, 2)] \
                + [(-2, 2), (-2, 2), (1, 2), (-2, 2)]*4
    p_name   = ['α_act', 'β_act', 'r'] \
                + get_param_name(['α+', 'α_', 'β', 'λ'])
    p_priors = [norm(0,1.5), norm(2, 1), norm(0,1.5)] + \
                [norm(0,1.5), norm(0,1.5), norm(2, 1), norm(0,1.5)]*4
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x), 
                lambda x: 1/(1+clip_exp(-x))] + \
               [lambda x: 1/(1+clip_exp(-x)),
                lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x), 
                lambda x: 1/(1+clip_exp(-x))]*4
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'alpha', 'valence', 'beta', 'lmbda'] 
    color    = viz.Gray
   
    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.beta_act       = params[1]
        self.r              = params[2]

        # ---- Stable & gain ---- #
        self.a_pos_sta_gain = params[3]
        self.a_neg_sta_gain = params[4]
        self.beta_sta_gain  = params[5]
        self.lmbda_sta_gain = params[6]

        # ---- Stable & loss ---- #
        self.a_pos_sta_loss = params[7]
        self.a_neg_sta_loss = params[8]
        self.beta_sta_loss  = params[9]
        self.lmbda_sta_loss = params[10]

        # ---- Volatile & gain ---- #
        self.a_pos_vol_gain = params[11]
        self.a_neg_vol_gain = params[12]
        self.beta_vol_gain  = params[13]
        self.lmbda_vol_gain = params[14]

        # ---- Volatile & loss ---- #
        self.a_pos_vol_loss = params[15]
        self.a_neg_vol_loss = params[16]
        self.beta_vol_loss  = params[17]
        self.lmbda_vol_loss = params[18]
    
    def learn(self):
        self._learn_critic()
        self._learn_actor()

    def _init_actor(self):
        self.q1     = 1/2
       
    def _learn_actor(self):
        a = self.mem.sample('a')
        self.q1 += self.alpha_act * (a - self.q1)
       
    def policy(self, m, **kwargs):
        t, f = kwargs['t_type'], kwargs['f_type']
        m0, m1 = m[0], m[1] 
        lmbda = eval(f'self.lmbda_{t}_{f}')
        v    = lmbda*(self.p_S[1] - self.p_S[0]) \
                + (1-lmbda)*abs(m1-m0)**self.r*np.sign(m1-m0)
        va   = eval(f'self.beta_{t}_{f}')*v \
                + self.beta_act*(self.q1 - (1-self.q1))
        pi1  = 1 / (1 + clip_exp(-va))
        self.pi = np.array([1-pi1, pi1])
        return self.pi  
    
    def get_beta(self):
        t, f = self.mem.sample('t_type', 'f_type')
        return eval(f'self.beta_{t}_{f}')
    
    def get_lmbda(self):
        t, f = self.mem.sample('t_type', 'f_type')
        return eval(f'self.lmbda_{t}_{f}')

class FLR22(FLR19):
    name     = 'FLR22'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (-2, 2)] \
                + [(-2, 2), (-2, 2), (1, 2), (-2, 2), (1, 2)]*4
    p_name   = ['α_act', 'r'] \
                + get_param_name(['α+', 'α_', 'β', 'λ', 'β_act'])
    p_priors = [norm(0,1.5), norm(0,1.5)] + \
                [norm(0,1.5), norm(0,1.5), norm(2, 1), norm(0,1.5), norm(2, 1)]*4
    p_trans  = [lambda x: 1/(1+clip_exp(-x)),
                lambda x: 1/(1+clip_exp(-x))] + \
               [lambda x: 1/(1+clip_exp(-x)),
                lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x), 
                lambda x: 1/(1+clip_exp(-x)),
                lambda x: clip_exp(x)]*4
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'alpha', 'valence', 'beta', 'lmbda'] 
    color    = viz.Gray

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.r              = params[1]

        # ---- Stable & gain ---- #
        self.a_pos_sta_gain = params[2]
        self.a_neg_sta_gain = params[3]
        self.beta_sta_gain  = params[4]
        self.lmbda_sta_gain = params[5]
        self.beta_act_sta_gain = params[6]

        # ---- Stable & loss ---- #
        self.a_pos_sta_loss = params[7]
        self.a_neg_sta_loss = params[8]
        self.beta_sta_loss  = params[9]
        self.lmbda_sta_loss = params[10]
        self.beta_act_sta_loss = params[11]

        # ---- Volatile & gain ---- #
        self.a_pos_vol_gain = params[12]
        self.a_neg_vol_gain = params[13]
        self.beta_vol_gain  = params[14]
        self.lmbda_vol_gain = params[15]
        self.beta_act_vol_gain = params[16]

        # ---- Volatile & loss ---- #
        self.a_pos_vol_loss = params[17]
        self.a_neg_vol_loss = params[18]
        self.beta_vol_loss  = params[19]
        self.lmbda_vol_loss = params[20]
        self.beta_act_vol_loss = params[21]

    def policy(self, m, **kwargs):
        t, f = kwargs['t_type'], kwargs['f_type']
        m0, m1 = m[0], m[1] 
        lmbda = eval(f'self.lmbda_{t}_{f}')
        beta_act = eval(f'self.beta_act_{t}_{f}')
        v     = lmbda*(self.p_S[1] - self.p_S[0]) \
                 + (1-lmbda)*abs(m1-m0)**self.r*np.sign(m1-m0)
        va    = eval(f'self.beta_{t}_{f}')*v \
                 + beta_act*(self.q1 - (1-self.q1))
        pi1   = 1 / (1 + clip_exp(-va))
        self.pi = np.array([1-pi1, pi1])
        return self.pi  

class FLR21(FLR19):
    name     = 'FLR22'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (-2, 2), (-2, 2), (1, 2), (1, 2)] \
                + [(-2, 2), (-2, 2), (1, 2), (-2, 2)]*4
    p_name   = ['α_act', 'r_gain', 'r_loss', 'β_act_gain', 'β_act_loss'] \
                + get_param_name(['α+', 'α_', 'β', 'λ'])
    p_priors = [norm(0,1.5), norm(2, 1), norm(2, 1), norm(2, 1), norm(2, 1)] + \
                [norm(0,1.5), norm(0,1.5), norm(2, 1), norm(0,1.5)]*4
    p_trans  = [lambda x: 1/(1+clip_exp(-x)),
                lambda x: clip_exp(x),
                lambda x: clip_exp(x), 
                lambda x: clip_exp(x), 
                lambda x: clip_exp(x)] + \
               [lambda x: 1/(1+clip_exp(-x)),
                lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x),
                lambda x: 1/(1+clip_exp(-x))]*4
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'alpha', 'valence', 'beta', 'lmbda'] 
    color    = viz.Gray

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha_act      = params[0]

        # ---- Gain & loss --- #
        self.r_gain         = params[1]
        self.r_loss         = params[2]
        self.beta_act_gain  = params[3]
        self.beta_act_loss  = params[4]

        # ---- parameters for each contxt ---- #
        i = 5
        for f in ['gain', 'loss']:
            for t in ['sta', 'vol']:
                for v in ['a_pos', 'a_neg', 'beta', 'lmbda']:
                    setattr(self, f'{v}_{t}_{f}', params[i])
                    i +=1
      
    def policy(self, m, **kwargs):
        t, f = kwargs['t_type'], kwargs['f_type']
        m0, m1 = m[0], m[1] 
        lmbda = eval(f'self.lmbda_{t}_{f}')
        r = eval(f'self.r_{f}')
        beta_act = eval(f'self.beta_act_{f}')
        v     = lmbda*(self.p_S[1] - self.p_S[0]) \
                 + (1-lmbda)*abs(m1-m0)**r*np.sign(m1-m0)
        va    = eval(f'self.beta_{t}_{f}')*v \
                 + beta_act*(self.q1 - (1-self.q1))
        pi1   = 1 / (1 + clip_exp(-va))
        self.pi = np.array([1-pi1, pi1])
        return self.pi  

class FLR6(FLR19):
    name     = 'FLR6'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2), (-2, 2), (-2, 2), (-2, 2), (1, 2)]
    p_name   = ['α_act', 'β_act', 'r', 'α', 'β', 'λ']
    p_priors = [norm(0,1.5), norm(2, 1), norm(0,1.5), norm(0,1.5), norm(2, 1), norm(0,1.5)]
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x), 
                lambda x: 1/(1+clip_exp(-x)),
                lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x), 
                lambda x: 1/(1+clip_exp(-x))]
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'alpha', 'valence', 'beta', 'lmbda'] 
    color    = viz.Gray
   
    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.beta_act       = params[1]
        self.r              = params[2]
        self.alpha_fix      = params[3]
        self.beta_fix       = params[4]
        self.lmbda_fix      = params[5]

        # ---- Stable & gain ---- #
        self.a_pos_sta_gain = self.alpha_fix
        self.a_neg_sta_gain = self.alpha_fix
        self.beta_sta_gain  = self.beta_fix
        self.lmbda_sta_gain = self.lmbda_fix

        # ---- Stable & loss ---- #
        self.a_pos_sta_loss = self.alpha_fix
        self.a_neg_sta_loss = self.alpha_fix
        self.beta_sta_loss  = self.beta_fix
        self.lmbda_sta_loss = self.lmbda_fix

        # ---- Volatile & gain ---- #
        self.a_pos_vol_gain = self.alpha_fix
        self.a_neg_vol_gain = self.alpha_fix
        self.beta_vol_gain  = self.beta_fix
        self.lmbda_vol_gain = self.lmbda_fix

        # ---- Volatile & loss ---- #
        self.a_pos_vol_loss = self.alpha_fix
        self.a_neg_vol_loss = self.alpha_fix
        self.beta_vol_loss  = self.beta_fix
        self.lmbda_vol_loss = self.lmbda_fix

# ------------------------------------------ #
#            Risk sensitive model            #
# ------------------------------------------ #

class RS13(RL):
    name     = 'RS13'
    p_bnds   = None
    p_pbnds  = [(1, 2)] + [(-2, 2), (-2, 2), (-2, 2)]*4
    p_name   = ['β'] + get_param_name(['α+', 'α-', 'γ'])
    n_params = len(p_name)
    p_priors = [norm(2, 1)] + [norm(0,1.5), norm(0,1.5), norm(2, 1)]*4
    p_trans  = [lambda x: clip_exp(x)] + \
               [lambda x: 1/(1+clip_exp(-x)),
                lambda x: 1/(1+clip_exp(-x)),
                lambda x: clip_exp(x)]*4
    voi      = ['pS1', 'pi1', 'valence', 'alpha'] 
    color    = viz.Gray

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.beta           = params[0]

        # ---- Stable & gain ---- #
        self.a_pos_sta_gain = params[1]
        self.a_neg_sta_gain = params[2]
        self.gamma_sta_gain = params[3]

        # ---- Stable & loss ---- #
        self.a_pos_sta_loss = params[4]
        self.a_neg_sta_loss = params[5]
        self.gamma_sta_loss = params[6]

        # ---- Volatile & gain ---- #
        self.a_pos_vol_gain = params[7]
        self.a_neg_vol_gain = params[8]
        self.gamma_vol_gain = params[9]

        # ---- Voatile & gain ---- #
        self.a_pos_vol_loss = params[10]
        self.a_neg_vol_loss = params[11]
        self.gamma_vol_loss = params[12]

    def _learn_critic(self):
        t, f, s = self.mem.sample('t_type', 'f_type', 's')
        delta = s-self.p1
        o = 'pos' if delta>0 else 'neg'
        self.o = o 
        self.p1 += eval(f'self.a_{o}_{t}_{f}') * delta 
        ps1 = np.clip(eval(f'self.gamma_{t}_{f}')*(self.p1-.5)+.5, 0, 1)
        self.p_S = np.array([1-ps1, ps1])

    def get_gamma(self):
        t, f = self.mem.sample('t_type', 'f_type')
        return eval(f'self.gamma_{t}_{f}') 

class RS3(RS13):
    name     = 'RS3'
    p_bnds   = None
    p_pbnds  = [(1, 2), (-2, -2), (-2, 2)]
    p_name   = ['β', 'α', 'γ']
    p_priors = [norm(2, 1), norm(0,1.5), norm(2, 1)]
    p_trans  = [lambda x: clip_exp(x),
                lambda x: 1/(1+clip_exp(-x)),
                lambda x: clip_exp(x)]
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'alpha'] 
    color    = viz.Gray

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.beta           = params[0]
        self.alpha_fix      = params[1]
        self.gamma_fix      = params[2]

        # ---- Stable & gain ---- #
        self.a_pos_sta_gain = self.alpha_fix
        self.a_neg_sta_gain = self.alpha_fix
        self.gamma_sta_gain = self.gamma_fix

        # ---- Stable & loss ---- #
        self.a_pos_sta_loss = self.alpha_fix
        self.a_neg_sta_loss = self.alpha_fix
        self.gamma_sta_loss = self.gamma_fix

        # ---- Volatile & gain ---- #
        self.a_pos_vol_gain = self.alpha_fix
        self.a_neg_vol_gain = self.alpha_fix
        self.gamma_vol_gain = self.gamma_fix

        # ---- Voatile & gain ---- #
        self.a_pos_vol_loss = self.alpha_fix
        self.a_neg_vol_loss = self.alpha_fix
        self.gamma_vol_loss = self.gamma_fix

# ------------------------------------------ #
#       Mixature of Strategies model         #
# ------------------------------------------ #

class EU(RL):
    name     = 'EU'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (-10, -.15)]
    p_name   = ['β', 'α']
    p_priors = []
    p_trans  = [lambda x: clip_exp(x), 
                lambda x: 1/(1+clip_exp(-x))]
    n_params = len(p_name)
    voi      = ['pS1', 'pi1']
    color    = viz.r2 
   
    def load_params(self, params):
        # assign the parameter
        self.beta           = params[0]
        self.alpha          = params[1]
        self.a_pos_sta_gain = self.alpha
        self.a_neg_sta_gain = self.alpha
        self.a_pos_sta_loss = self.alpha
        self.a_neg_sta_gain = self.alpha
        self.a_pos_vol_gain = self.alpha
        self.a_neg_vol_gain = self.alpha
        self.a_pos_vol_loss = self.alpha
        self.a_neg_vol_loss = self.alpha

    def policy(self, m, **kwargs):
        self.pi = softmax(self.beta*self.p_S*m)
        return self.pi
    
class PS(EU):
    name     = 'PF'
   
    def policy(self, m, **kwargs):
        self.pi = softmax(self.beta*self.p_S)
        return self.pi

class MOS22(RL):
    name     = 'MOS22'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2)] + ([(-2, 2)]*2+[(-6, 6)]*3) * 4
    p_name   = ['α_act', 'β'] + get_param_name(['α+', 'α_', 'λ1', 'λ2', 'λ3'])
    p_priors = [norm(0,1.5), norm(2, 1)]+\
                [norm(0, 1.5), norm(0, 1.5), 
                norm(0, 10), norm(0, 10), norm(0, 10)]*4
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x)] \
                + ([lambda x: 1/(1+clip_exp(-x)),
                    lambda x: 1/(1+clip_exp(-x))]+
                   [lambda x: x]*3) * 4 
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence', 'alpha', 'l1', 'l2', 'l3', 'w1', 'w2', 'w3']
    color    = viz.r2

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.beta           = params[1]

        # ---- Stable & gain ---- #
        self.a_pos_sta_gain = params[2]
        self.a_neg_sta_gain = params[3]
        self.l0_sta_gain    = params[4]
        self.l1_sta_gain    = params[5]
        self.l2_sta_gain    = params[6]

        # ---- Stable & loss ---- #
        self.a_pos_sta_loss = params[7]
        self.a_neg_sta_loss = params[8]
        self.l0_sta_loss    = params[9]
        self.l1_sta_loss    = params[10]
        self.l2_sta_loss    = params[11]

        # ---- Volatile & gain ---- #
        self.a_pos_vol_gain = params[12]
        self.a_neg_vol_gain = params[13]
        self.l0_vol_gain    = params[14]
        self.l1_vol_gain    = params[15]
        self.l2_vol_gain    = params[16]

        # ---- Volatile & loss ---- #
        self.a_pos_vol_loss = params[17]
        self.a_neg_vol_loss = params[18]
        self.l0_vol_loss    = params[19]
        self.l1_vol_loss    = params[20]
        self.l2_vol_loss    = params[21]    
    
    def _init_actor(self):
        self.q1  = .5
        self.q_A = np.array([1-self.q1, self.q1]) 
        self.pi_effect = [1/3, 1/3, 1/3]
        self.pi  = np.array([.5, .5])

    #  ------ learning the probability ------- #

    def learn(self):
        self._learn_critic()
        self._learn_actor()

    def _learn_actor(self):
        a = self.mem.sample('a')
        self.q1 += self.alpha_act * (a - self.q1)
        self.q_A = np.array([1-self.q1, self.q1])

    #  ------ response strategy ------- #

    def policy(self, m, **kwargs):
        t, f = kwargs['t_type'], kwargs['f_type']
        pi_SM = softmax(self.beta*self.p_S*m)
        pi_M  = softmax(self.beta*m)
        w0, w1, w2 = self.get_w(t, f)
        # creat the mixature model 
        self.pi_effect = [pi_SM[1], pi_M[1], self.q_A[1]]
        self.pi = w0*pi_SM + w1*pi_M + w2*self.q_A 
        return self.pi 

    def get_w(self, b, f):
        l0 = eval(f'self.l0_{b}_{f}')
        l1 = eval(f'self.l1_{b}_{f}')
        l2 = eval(f'self.l2_{b}_{f}')
        return softmax([l0, l1, l2])

    #  ------ print variable of interests ------- #

    def get_w1(self):
        t, f = self.mem.sample('t_type', 'f_type')
        return self.get_w(t, f)[0]

    def get_w2(self):
        t, f = self.mem.sample('t_type', 'f_type')
        return self.get_w(t, f)[1]

    def get_w3(self):
        t, f = self.mem.sample('t_type', 'f_type')
        return self.get_w(t, f)[2]  

    def get_l1(self):
        t, f = self.mem.sample('t_type', 'f_type')
        return eval(f'self.l0_{t}_{f}')

    def get_l2(self):
        t, f = self.mem.sample('t_type', 'f_type')
        return eval(f'self.l1_{t}_{f}')

    def get_l3(self):
        t, f = self.mem.sample('t_type', 'f_type')
        return eval(f'self.l2_{t}_{f}')

    def get_l1_effect(self):
        return self.pi_effect[0]

    def get_l2_effect(self):
        return self.pi_effect[1]

    def get_l3_effect(self):
        return self.pi_effect[2]

class EU_MO18(MOS22):
    name     = 'EU+MO18'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2)] + ([(-2, 2)]*2+[(-6, 6)]*2) * 4
    p_name   = ['α_act', 'β'] + get_param_name(['α+', 'α_', 'λ1', 'λ2'])
    p_priors = [norm(0,1.5), norm(2, 1)]+\
                [norm(0, 1.5), norm(0, 1.5), norm(0, 10), norm(0, 10)]*4
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x)] \
                + ([lambda x: 1/(1+clip_exp(-x)),
                    lambda x: 1/(1+clip_exp(-x))]+
                   [lambda x: x]*2) * 4 
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence', 'alpha', 'l1', 'l2']
    color    = viz.Pizazz

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.beta           = params[1]

        # ---- Stable & gain ---- #
        self.a_pos_sta_gain = params[2]
        self.a_neg_sta_gain = params[3]
        self.l0_sta_gain    = params[4]
        self.l1_sta_gain    = params[5]
        self.l2_sta_gain    = -max_ #params[6]

        # ---- Stable & loss ---- #
        self.a_pos_sta_loss = params[6]
        self.a_neg_sta_loss = params[7]
        self.l0_sta_loss    = params[8]
        self.l1_sta_loss    = params[9]
        self.l2_sta_loss    = -max_ #params[11]

        # ---- Volatile & gain ---- #
        self.a_pos_vol_gain = params[10]
        self.a_neg_vol_gain = params[11]
        self.l0_vol_gain    = params[12]
        self.l1_vol_gain    = params[13]
        self.l2_vol_gain    = -max_ #params[16]

        # ---- Volatile & loss ---- #
        self.a_pos_vol_loss = params[14]
        self.a_neg_vol_loss = params[15]
        self.l0_vol_loss    = params[16]
        self.l1_vol_loss    = params[17]
        self.l2_vol_loss    = -max_ #params[21]    

class EU_HA18(MOS22):
    name     = 'EU+HA18'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2)] + ([(-2, 2)]*2+[(-6, 6)]*2) * 4
    p_name   = ['α_act', 'β'] + get_param_name(['α+', 'α_', 'λ1', 'λ3'])
    p_priors = [norm(0,1.5), norm(2, 1)]+\
                [norm(0, 1.5), norm(0, 1.5), 
                 norm(0, 10), norm(0, 10)]*4
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x)] \
                + ([lambda x: 1/(1+clip_exp(-x)),
                    lambda x: 1/(1+clip_exp(-x))]+
                   [lambda x: x]*2) * 4 
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence', 'alpha', 'l1', 'l3']
    color    = viz.Pizazz

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.beta           = params[1]

        # ---- Stable & gain ---- #
        self.a_pos_sta_gain = params[2]
        self.a_neg_sta_gain = params[3]
        self.l0_sta_gain    = params[4]
        self.l1_sta_gain    = -max_ 
        self.l2_sta_gain    = params[5]

        # ---- Stable & loss ---- #
        self.a_pos_sta_loss = params[6]
        self.a_neg_sta_loss = params[7]
        self.l0_sta_loss    = params[8]
        self.l1_sta_loss    = -max_ 
        self.l2_sta_loss    = params[9]

        # ---- Volatile & gain ---- #
        self.a_pos_vol_gain = params[10]
        self.a_neg_vol_gain = params[11]
        self.l0_vol_gain    = params[12]
        self.l1_vol_gain    = -max_ 
        self.l2_vol_gain    = params[13]

        # ---- Volatile & loss ---- #
        self.a_pos_vol_loss = params[14]
        self.a_neg_vol_loss = params[15]
        self.l0_vol_loss    = params[16]
        self.l1_vol_loss    = -max_ 
        self.l2_vol_loss    = params[17]    

class MO_HA18(MOS22):
    name     = 'MO+HA18'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2)] + ([(-2, 2)]*2+[(-6, 6)]*2) * 4
    p_name   = ['α_act', 'β'] + get_param_name(['α+', 'α_', 'λ2', 'λ3'])
    p_priors = [norm(0,1.5), norm(2, 1)]+\
                [norm(0, 1.5), norm(0, 1.5), 
                 norm(0, 10), norm(0, 10)]*4
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x)] \
                + ([lambda x: 1/(1+clip_exp(-x)),
                    lambda x: 1/(1+clip_exp(-x))]+
                   [lambda x: x]*2) * 4 
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence', 'alpha', 'l2', 'l3']
    color    = viz.Pizazz

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.beta           = params[1]

        # ---- Stable & gain ---- #
        self.a_pos_sta_gain = params[2]
        self.a_neg_sta_gain = params[3]
        self.l0_sta_gain    = -max_
        self.l1_sta_gain    = params[4] 
        self.l2_sta_gain    = params[5]

        # ---- Stable & loss ---- #
        self.a_pos_sta_loss = params[6]
        self.a_neg_sta_loss = params[7]
        self.l0_sta_loss    = -max_
        self.l1_sta_loss    = params[8] 
        self.l2_sta_loss    = params[9]

        # ---- Volatile & gain ---- #
        self.a_pos_vol_gain = params[10]
        self.a_neg_vol_gain = params[11]
        self.l0_vol_gain    = -max_
        self.l1_vol_gain    = params[12] 
        self.l2_vol_gain    = params[13]

        # ---- Volatile & loss ---- #
        self.a_pos_vol_loss = params[14]
        self.a_neg_vol_loss = params[15]
        self.l0_vol_loss    = -max_
        self.l1_vol_loss    = params[16] 
        self.l2_vol_loss    = params[17]    

class PS_MO_HA22(MOS22):
    name     = 'PS+MO+HA22'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2)] + ([(-2, 2)]*2+[(-6, 6)]*3) * 4
    p_name   = ['α_act', 'β'] + get_param_name(['α+', 'α_', 'λ1', 'λ2', 'λ3'])
    p_priors = [norm(0,1.5), norm(2, 1)]+\
                [norm(0, 1.5), norm(0, 1.5), 
                norm(0, 10), norm(0, 10), norm(0, 10)]*4
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x)] \
                + ([lambda x: 1/(1+clip_exp(-x)),
                    lambda x: 1/(1+clip_exp(-x))]+
                   [lambda x: x]*3) * 4 
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence', 'alpha', 'l1', 'l2', 'l3']
    color    = viz.Green

    def policy(self, m, **kwargs):
        t, f = kwargs['t_type'], kwargs['f_type']
        pi_S = softmax(self.beta*self.p_S)
        pi_M  = softmax(self.beta*m)
        w0, w1, w2 = self.get_w(t, f)
        # creat the mixature model 
        self.pi_effect = [pi_S[1], pi_M[1], self.q_A[1]]
        self.pi = w0*pi_S + w1*pi_M + w2*self.q_A 
        return self.pi 

class EU_PS_MO_HA26(MOS22):
    name     = 'EU+PS+MO+HA26'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2)] + ([(-2, 2)]*2+[(-6, 6)]*4) * 4
    p_name   = ['α_act', 'β'] + get_param_name(['α+', 'α_', 'λ1', 'λ2', 'λ3', 'λ4'])
    p_priors = [norm(0,1.5), norm(2, 1)]+\
                [norm(0, 1.5), norm(0, 1.5), 
                norm(0, 10), norm(0, 10), norm(0, 10), norm(0, 10)]*4
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x)] \
                + ([lambda x: 1/(1+clip_exp(-x)),
                    lambda x: 1/(1+clip_exp(-x))]+
                   [lambda x: x]*4) * 4 
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence', 'alpha', 'l1', 'l2', 'l3', 'l4']
    color    = viz.SteelBlu

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.beta           = params[1]

        # ---- Stable & gain ---- #
        self.a_pos_sta_gain = params[2]
        self.a_neg_sta_gain = params[3]
        self.l0_sta_gain    = params[4]
        self.l1_sta_gain    = params[5]
        self.l2_sta_gain    = params[6]
        self.l3_sta_gain    = params[7]

        # ---- Stable & loss ---- #
        self.a_pos_sta_loss = params[8]
        self.a_neg_sta_loss = params[9]
        self.l0_sta_loss    = params[10]
        self.l1_sta_loss    = params[11]
        self.l2_sta_loss    = params[12]
        self.l3_sta_loss    = params[13]

        # ---- Volatile & gain ---- #
        self.a_pos_vol_gain = params[14]
        self.a_neg_vol_gain = params[15]
        self.l0_vol_gain    = params[16]
        self.l1_vol_gain    = params[17]
        self.l2_vol_gain    = params[18]
        self.l3_vol_gain    = params[19]

        # ---- Volatile & loss ---- #
        self.a_pos_vol_loss = params[20]
        self.a_neg_vol_loss = params[21]
        self.l0_vol_loss    = params[22]
        self.l1_vol_loss    = params[23]
        self.l2_vol_loss    = params[24]    
        self.l3_vol_loss    = params[25]
  
    #  ------ response strategy ------- #

    def policy(self, m, **kwargs):
        t, f = kwargs['t_type'], kwargs['f_type']
        pi_SM = softmax(self.beta*self.p_S*m)
        pi_S  = softmax(self.beta*self.p_S)
        pi_M  = softmax(self.beta*m)
        w0, w1, w2, w3 = self.get_w(t, f)
        # creat the mixature model 
        self.pi_effect = [pi_SM[1], pi_S[1], pi_M[1], self.q_A[1]]
        self.pi = w0*pi_SM + w1*pi_S + w2*pi_M + w3*self.q_A 
        return self.pi 

    def get_w(self, b, f):
        l0 = eval(f'self.l0_{b}_{f}')
        l1 = eval(f'self.l1_{b}_{f}')
        l2 = eval(f'self.l2_{b}_{f}')
        l3 = eval(f'self.l3_{b}_{f}')
        return softmax([l0, l1, l2, l3])

    def get_w4(self):
        t, f = self.mem.sample('t_type', 'f_type')
        return self.get_w(t, f)[3]  

    def get_l4(self):
        t, f = self.mem.sample('t_type', 'f_type')
        return eval(f'self.l3_{t}_{f}')

    def get_l4_effect(self):
        return self.pi_effect[3]

# a set of parameters for all contexts

class MOS6(MOS22):
    name     = 'MOS6'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2), (-2, 2)] + [(-6, 6)]*3
    p_name   = ['α_act', 'β', 'α', 'λ1', 'λ2', 'λ3']
    p_priors = [norm(0,1.5), norm(2, 1), norm(0,1.5), 
                norm(0, 10), norm(0, 10), norm(0, 10)]
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x),
                lambda x: 1/(1+clip_exp(-x))] \
                + [lambda x: x]*3
    p_poi    = ['α', 'λ1', 'λ2', 'λ3']
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence', 'alpha', 'beta', 'alpha_act', 'l1', 'l2', 'l3']
    color    = viz.r1

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.beta           = params[1]
        self.alpha_fix      = params[2]
        self.l0_fix         = params[3]
        self.l1_fix         = params[4]
        self.l2_fix         = params[5]
        self.assign_fix_val()

    def assign_fix_val(self):

        # ---- Stable & gain ---- #
        self.a_pos_sta_gain = self.alpha_fix
        self.a_neg_sta_gain = self.alpha_fix
        self.l0_sta_gain    = self.l0_fix
        self.l1_sta_gain    = self.l1_fix
        self.l2_sta_gain    = self.l2_fix

        # ---- Stable & loss ---- #
        self.a_pos_sta_loss = self.alpha_fix
        self.a_neg_sta_loss = self.alpha_fix
        self.l0_sta_loss    = self.l0_fix
        self.l1_sta_loss    = self.l1_fix
        self.l2_sta_loss    = self.l2_fix

        # ---- Volatile & gain ---- #
        self.a_pos_vol_gain = self.alpha_fix
        self.a_neg_vol_gain = self.alpha_fix
        self.l0_vol_gain    = self.l0_fix
        self.l1_vol_gain    = self.l1_fix
        self.l2_vol_gain    = self.l2_fix

        # ---- Volatile & loss ---- #
        self.a_pos_vol_loss = self.alpha_fix
        self.a_neg_vol_loss = self.alpha_fix
        self.l0_vol_loss    = self.l0_fix
        self.l1_vol_loss    = self.l1_fix
        self.l2_vol_loss    = self.l2_fix

    def get_beta(self): return self.beta
    
    def get_alpha_act(self): return self.alpha_act

class EU_MO(MOS6):
    name     = 'EU+MO'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2), (-2, 2)] + [(-6, 6)]*2
    p_name   = ['α_act', 'β', 'α', 'λ1', 'λ2']
    p_priors = [norm(0,1.5), norm(2, 1), norm(0,1.5), 
                norm(0, 10), norm(0, 10)]
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x),
                lambda x: 1/(1+clip_exp(-x))] \
                + [lambda x: x]*2
    p_poi    = ['α', 'λ1', 'λ2']
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence', 'alpha', 'l1', 'l2']
    color    = viz.Pizazz

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.beta           = params[1]
        self.alpha_fix      = params[2]
        self.l0_fix         = params[3]
        self.l1_fix         = params[4]
        # a large negative number to ensure that the 
        # Softmax weight is 0. exp(-max_) = 0
        self.l2_fix         = -max_ 
        self.assign_fix_val()
    
class EU_HA(MOS6):
    name     = 'EU+HA'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2), (-2, 2)] + [(-6, 6)]*2
    p_name   = ['α_act', 'β', 'α', 'λ1', 'λ3']
    p_priors = [norm(0,1.5), norm(2, 1), norm(0,1.5), 
                norm(0, 10), norm(0, 10)]
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x),
                lambda x: 1/(1+clip_exp(-x))] \
                + [lambda x: x]*2
    p_poi    = ['α', 'λ1', 'λ3']
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence', 'alpha', 'l1', 'l3']
    color    = viz.Pizazz

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.beta           = params[1]
        self.alpha_fix      = params[2]
        # a large negative number to ensure that the 
        # Softmax weight is 0. exp(-max_) = 0
        self.l0_fix         = params[3]
        self.l1_fix         = -max_
        self.l2_fix         = params[4] 
        self.assign_fix_val()

class MO_HA(MOS6):
    name     = 'MO+HA'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2), (-2, 2)] + [(-6, 6)]*2
    p_name   = ['α_act', 'β', 'α', 'λ2', 'λ3']
    p_priors = [norm(0,1.5), norm(2, 1), norm(0,1.5), 
                norm(0, 10), norm(0, 10)]
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x),
                lambda x: 1/(1+clip_exp(-x))] \
                + [lambda x: x]*2
    p_poi    = ['α', 'λ2', 'λ3']
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence', 'alpha', 'l2', 'l3']
    color    = viz.Pizazz

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.beta           = params[1]
        self.alpha_fix      = params[2]
        # a large negative number to ensure that the 
        # Softmax weight is 0. exp(-max_) = 0
        self.l0_fix         = -max_
        self.l1_fix         = params[3]
        self.l2_fix         = params[4] 
        self.assign_fix_val()

class PS_MO_HA(MOS6):
    name     = 'PF+MO+HA'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2), (-2, 2)] + [(-6, 6)]*3
    p_name   = ['α_act', 'β', 'α', 'λ1', 'λ2', 'λ3']
    p_priors = [norm(0,1.5), norm(2, 1), norm(0,1.5), 
                norm(0, 10), norm(0, 10), norm(0, 10)]
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x),
                lambda x: 1/(1+clip_exp(-x))] \
                + [lambda x: x]*3
    p_poi    = ['α', 'λ1', 'λ2', 'λ3']
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence', 'alpha', 'l1', 'l2', 'l3']
    color    = viz.Green

    def policy(self, m, **kwargs):
        t, f = kwargs['t_type'], kwargs['f_type']
        pi_S = softmax(self.beta*self.p_S)
        pi_M  = softmax(self.beta*m)
        w0, w1, w2 = self.get_w(t, f)
        # creat the mixature model 
        self.pi_effect = [pi_S[1], pi_M[1], self.q_A[1]]
        self.pi = w0*pi_S + w1*pi_M + w2*self.q_A 
        return self.pi 

class EU_PS_MO_HA(MOS6):
    name     = 'EU+PF+MO+HA'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2), (-2, 2)] + [(-6, 6)]*4
    p_name   = ['α_act', 'β', 'α', 'λ1', 'λ2', 'λ3', 'λ4']
    p_priors = [norm(0,1.5), norm(2, 1), norm(0,1.5), 
                norm(0, 10), norm(0, 10), norm(0, 10), norm(0, 10)]
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x),
                lambda x: 1/(1+clip_exp(-x))] \
                + [lambda x: x]*4
    p_poi    = ['α', 'λ1', 'λ2', 'λ3', 'λ4']
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence', 'alpha', 'l1', 'l2', 'l3', 'l4']
    color    = viz.SteelBlu

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.beta           = params[1]
        self.alpha_fix      = params[2]
        self.l0_fix         = params[3]
        self.l1_fix         = params[4]
        self.l2_fix         = params[5]
        self.l3_fix         = params[6]
  
        # ---- Stable & gain ---- #
        self.a_pos_sta_gain = self.alpha_fix
        self.a_neg_sta_gain = self.alpha_fix
        self.l0_sta_gain    = self.l0_fix
        self.l1_sta_gain    = self.l1_fix
        self.l2_sta_gain    = self.l2_fix
        self.l3_sta_gain    = self.l3_fix

        # ---- Stable & loss ---- #
        self.a_pos_sta_loss = self.alpha_fix
        self.a_neg_sta_loss = self.alpha_fix
        self.l0_sta_loss    = self.l0_fix
        self.l1_sta_loss    = self.l1_fix
        self.l2_sta_loss    = self.l2_fix
        self.l3_sta_loss    = self.l3_fix

        # ---- Volatile & gain ---- #
        self.a_pos_vol_gain = self.alpha_fix
        self.a_neg_vol_gain = self.alpha_fix
        self.l0_vol_gain    = self.l0_fix
        self.l1_vol_gain    = self.l1_fix
        self.l2_vol_gain    = self.l2_fix
        self.l3_vol_gain    = self.l3_fix

        # ---- Volatile & loss ---- #
        self.a_pos_vol_loss = self.alpha_fix
        self.a_neg_vol_loss = self.alpha_fix
        self.l0_vol_loss    = self.l0_fix
        self.l1_vol_loss    = self.l1_fix
        self.l2_vol_loss    = self.l2_fix
        self.l3_vol_loss    = self.l3_fix   
  
    #  ------ response strategy ------- #

    def policy(self, m, **kwargs):
        t, f = kwargs['t_type'], kwargs['f_type']
        pi_SM = softmax(self.beta*self.p_S*m)
        pi_S  = softmax(self.beta*self.p_S)
        pi_M  = softmax(self.beta*m)
        w0, w1, w2, w3 = self.get_w(t, f)
        # creat the mixature model 
        self.pi_effect = [pi_SM[1], pi_S[1], pi_M[1], self.q_A[1]]
        self.pi = w0*pi_SM + w1*pi_S + w2*pi_M + w3*self.q_A 
        return self.pi 

    def get_w(self, b, f):
        l0 = eval(f'self.l0_{b}_{f}')
        l1 = eval(f'self.l1_{b}_{f}')
        l2 = eval(f'self.l2_{b}_{f}')
        l3 = eval(f'self.l3_{b}_{f}')
        return softmax([l0, l1, l2, l3])

    def get_w4(self):
        t, f = self.mem.sample('t_type', 'f_type')
        return self.get_w(t, f)[3]  

    def get_l4(self):
        t, f = self.mem.sample('t_type', 'f_type')
        return eval(f'self.l3_{t}_{f}')

    def get_l4_effect(self):
        return self.pi_effect[3]

class EU_MO_HA_RD(EU_PS_MO_HA):
    name     = 'EU+MO+HA+RD'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2), (-2, 2)] + [(-6, 6)]*4
    p_name   = ['α_act', 'β', 'α', 'λ1', 'λ2', 'λ3', 'λ4']
    p_priors = [norm(0,1.5), norm(2, 1), norm(0,1.5), 
                norm(0, 10), norm(0, 10), norm(0, 10), norm(0, 10)]
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x),
                lambda x: 1/(1+clip_exp(-x))] \
                + [lambda x: x]*4
    p_poi    = ['α', 'λ1', 'λ2', 'λ3', 'λ4']
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence', 'alpha', 'l1', 'l2', 'l3', 'l4']
    color    = viz.SteelBlu

    #  ------ response strategy ------- #

    def policy(self, m, **kwargs):
        t, f = kwargs['t_type'], kwargs['f_type']
        pi_SM = softmax(self.beta*self.p_S*m)
        pi_M  = softmax(self.beta*m)
        pi_RD = np.ones([2,]) / 2
        w0, w1, w2, w3 = self.get_w(t, f)
        # creat the mixature model 
        self.pi_effect = [pi_SM[1], pi_M[1], self.q_A[1], pi_RD[1]]
        self.pi = w0*pi_SM + w1*pi_M + w2*self.q_A + w3*pi_RD
        return self.pi 

class linear_comb(MOS6):
    name     = 'linear comb.'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (1, 2), (-2, 2)] + [(-6, 6)]*3
    p_name   = ['α_act', 'β', 'α', 'λ1', 'λ2', 'λ3']
    p_priors = [norm(0,1.5), norm(2, 1), norm(0,1.5), 
                norm(0, 10), norm(0, 10), norm(0, 10)]
    p_trans  = [lambda x: 1/(1+clip_exp(-x)), 
                lambda x: clip_exp(x),
                lambda x: 1/(1+clip_exp(-x))] \
                + [lambda x: x]*3
    p_poi    = ['α', 'λ1', 'λ2', 'λ3']
    n_params = len(p_name)
    voi      = ['pS1', 'pi1', 'valence', 'alpha', 'l1', 'l2', 'l3']
    color    = viz.Green

    def policy(self, m, **kwargs):
        t, f = kwargs['t_type'], kwargs['f_type']
        pi_S = self.p_S
        pi_M = m
        l0, l1, l2 = self.get_w(t, f)
        # creat the mixature model 
        self.pi_effect = [pi_S[1], pi_M[1], self.q_A[1]]
        S_diff  = self.p_S[1] - self.p_S[0] 
        M_diff  = m[1] - m[0]
        HA_diff = self.q_A[1] - self.q_A[0] 
        diff = l0*S_diff + l1*M_diff + l2*HA_diff 
        pi1 = sigmoid(self.beta*diff)
        self.pi = np.array([1-pi1, pi1])
        return self.pi

    def get_w(self, b, f):
        l0 = eval(f'self.l0_{b}_{f}')
        l1 = eval(f'self.l1_{b}_{f}')
        l2 = eval(f'self.l2_{b}_{f}')
        return [l0, l1, l2]

# ------------------------------------------ #
#            Pearce Hall model               #
# ------------------------------------------ #

class PH17(RL):
    name     = 'PH17'
    p_bnds   = None
    p_pbnds  = [(-2, 2),] + [(-2, 2), (-2, 2), (-2, 2), (1, 2)]*4
    p_name   = ['α0'] + get_param_name(['k+', 'k_', 'η', 'β'])
    n_params = len(p_name)
    p_priors = [norm(0,1.5)]+\
                [norm(0,1.5), norm(0,1.5), 
                 norm(0,1.5), norm(2, 1)]*4
    p_trans  = [lambda x: 1/(1+clip_exp(-x))] + \
               [lambda x: 1/(1+clip_exp(-x)),
                lambda x: 1/(1+clip_exp(-x)),
                lambda x: 1/(1+clip_exp(-x)),
                lambda x: clip_exp(x)]*4
    voi      = ['pS1', 'pi1', 'alpha', 'valence']
    color    = viz.Gray

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha          = params[0]

        # ---- Stable & gain ---- #
        self.k_pos_sta_gain = params[1]
        self.k_neg_sta_gain = params[2]
        self.eta_sta_gain   = params[3]
        self.beta_sta_gain  = params[4]

        # ---- Stable & loss ---- #
        self.k_pos_sta_loss = params[5]
        self.k_neg_sta_loss = params[6]
        self.eta_sta_loss   = params[7]
        self.beta_sta_loss  = params[8]

        # ---- Volatile & gain ---- #
        self.k_pos_vol_gain = params[9]
        self.k_neg_vol_gain = params[10]
        self.eta_vol_gain   = params[11]
        self.beta_vol_gain  = params[12]


        # ---- Volatile & loss ---- #
        self.k_pos_vol_loss = params[13]
        self.k_neg_vol_loss = params[14]
        self.eta_vol_loss   = params[15]
        self.beta_vol_loss  = params[16]

    def learn(self):
        self._learn_critic()
        self._learn_alpha()

    def _learn_critic(self):
        s, t, f = self.mem.sample('s', 't_type', 'f_type')
        self.delta = s - self.p1
        o = 'pos' if self.delta>0 else 'neg'
        k = eval(f'self.k_{o}_{t}_{f}')
        self.p1 += k*self.alpha*self.delta
        self.p_S = np.array([1-self.p1, self.p1])

    def _learn_alpha(self,):
        t, f = self.mem.sample('t_type', 'f_type')
        eta = eval(f'self.eta_{t}_{f}')
        self.alpha += eta*(np.abs(self.delta)-self.alpha)

    def policy(self, m, **kwargs):
        t, f = kwargs['t_type'], kwargs['f_type']
        beta = eval(f'self.beta_{t}_{f}')
        self.pi = softmax(beta*self.p_S*m)
        return self.pi
    
    def get_alpha(self):
        o, t, f = self.mem.sample('o_type', 't_type', 'f_type')
        return eval(f'self.k_{o}_{t}_{f}')*self.alpha

class PH4(PH17):
    name     = 'PH4'
    p_bnds   = None
    p_pbnds  = [(-2, 2), (-2, 2), (-2, 2), (1, 2)]
    p_name   = ['α0', 'k', 'η', 'β']
    n_params = len(p_name)
    p_priors = [norm(0, 1.5), norm(0, 1.5), norm(0, 1.5), norm(2, 1)]
    p_trans  = [lambda x: 1/(1+clip_exp(-x)),
                lambda x: 1/(1+clip_exp(-x)),
                lambda x: 1/(1+clip_exp(-x)),
                lambda x: clip_exp(x)]
    voi      = ['pS1', 'pi1', 'alpha', 'valence']
    color    = viz.Gray

    def load_params(self, params):

        # from gauss space to actual space
        params = [fn(p) for fn, p in zip(self.p_trans, params)]

        # ---- General ----- #
        self.alpha          = params[0]
        self.k_fix          = params[1]
        self.eta_fix        = params[2]
        self.beta_fix       = params[3]

        # ---- Stable & gain ---- #
        self.k_pos_sta_gain = self.k_fix
        self.k_neg_sta_gain = self.k_fix
        self.eta_sta_gain   = self.eta_fix
        self.beta_sta_gain  = self.beta_fix

        # ---- Stable & loss ---- #
        self.k_pos_sta_loss = self.k_fix
        self.k_neg_sta_loss = self.k_fix
        self.eta_sta_loss   = self.eta_fix
        self.beta_sta_loss  = self.beta_fix

        # ---- Volatile & gain ---- #
        self.k_pos_vol_gain = self.k_fix
        self.k_neg_vol_gain = self.k_fix
        self.eta_vol_gain   = self.eta_fix
        self.beta_vol_gain  = self.beta_fix

        # ---- Volatile & loss ---- #
        self.k_pos_vol_loss = self.k_fix
        self.k_neg_vol_loss = self.k_fix
        self.eta_vol_loss   = self.eta_fix
        self.beta_vol_loss  = self.beta_fix
