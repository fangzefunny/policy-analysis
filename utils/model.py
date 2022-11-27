import numpy as np 
import pandas as pd 
from scipy.optimize import minimize

eps_ = 1e-13
max_ = 1e+13

class model:
    '''Out loop of the fit
    This class can instantiate a dynmaic decision-making model. 
    Two main functions:
        fit:  search for the best parameters given label
            arg max_{θ} p(x,y|θ)
        pred: predict the label given parameters 
            y ~ p(Y|x,θ)
    '''

    def __init__(self, agent):
        self.agent = agent
        self.param_priors = agent.p_priors
    
    # ------------ fit ------------ #

    def fit(self, data, method='mle', seed=2021, init=None, verbose=False):
        '''Fit the parameter using optimization 
        '''
        # get bounds and possible bounds 
        bnds  = self.agent.bnds
        pbnds = self.agent.pbnds
        if method == 'mle': self.param_priors = None 
        fit_method = 'L-BFGS-B' if method == 'bms' else 'Nelder-Mead'

        # Init params
        if init:
            # if there are assigned params
            param0 = init
        else:
            # random init from the possible bounds 
            rng = np.random.RandomState(seed)
            param0 = [pbnd[0] + (pbnd[1] - pbnd[0]
                     ) * rng.rand() for pbnd in pbnds]
                     
        ## Fit the params 
        if verbose: print('init with params: ', param0) 
        res = minimize(self.loss_fn, param0, args=(data), method=fit_method,
                        bounds=bnds, options={'disp': verbose})
        if verbose: print(f'''  Fitted params: {res.x}, 
                    MLE loss: {res.fun}''')
        
        return res

    def loss_fn(self, params, data):
        '''Total likelihood
        log p(D|θ) = -log ∏_i p(D_i|θ)
                   = ∑_i -log p(D_i|θ )
        or Maximum a posterior 
        log p(θ|D) = ∑_i -log p(D_i|θ ) + -log p(θ)
        '''
        tot_loss = [self._negloglike(params, data[key])
                  + self._neglogprior(params) 
                    for key in data.keys()]        

        return np.sum(tot_loss)

    def loglike(self, params, data):
        tot_loglike = [-self._negloglike(params, data[key])
                    for key in data.keys()]  
        return np.sum(tot_loglike) 

    def _negloglike(self, params, block_data):
        '''Likelihood for one sample
        -log p(D_i|θ )
        In RL, each sample is a block of experiment,
        Because it is independent across experiment.
        '''
        nA = block_data['state'].unique().shape[0]
        subj = self.agent(nA, params)
        nLL = 0
       
        ## loop to simulate the responses in the block 
        for _, row in block_data.iterrows():

            # predict stage: obtain input
            mag0   = row['mag0']
            mag1   = row['mag1']
            b_type = row['b_type']
            f_type = row['feedback_type']
            state  = row['state']
            act    = row['humanAct']
            # rew   = row['rew']
            mem  = {'b_type': b_type, 'f_type': f_type,
                    'mag0': mag0, 'mag1': mag1}
            subj.buffer.push(mem)

            # control stage: evaluate the human act
            nLL -= subj.control(act, mode='eval')

            # feedback stage: update the belief, 'gen' has no feedback
            mem = {'b_type': b_type, 'f_type': f_type,
                    'state': state, 'act': act}
            subj.buffer.push(mem)  
            subj.learn() 

        return nLL
          
    def _neglogprior(self, params):
        '''Add the prior of the parameters
        '''
        tot_pr = 0.
        if self.param_priors:
            for prior, param in zip(self.param_priors, params):
                tot_pr += -np.max([prior.logpdf(param), -max_])
        return tot_pr

    # ------------ simulate ------------ #

    def sim(self, data, params, rng=None):
        sim_data = [] 
        for block_id in data.keys():
            block_data = data[block_id].copy()
            try:
                block_data = block_data.drop(columns=['rew'])
            except:
                pass 
            sim_data.append(self.sim_block(block_data, params, rng))
        
        return pd.concat(sim_data, ignore_index=True)

    def sim_block(self, block_data, params, rng=False, is_eval=True):

        ## init the agent 
        nA = block_data['state'].unique().shape[0]
        subj = self.agent(nA, params)

        ## init a blank dataframe to store simulation
        col = ['act', 'match', 'acc', 'logLike'] + self.agent.voi
        init_mat = np.zeros([block_data.shape[0], len(col)]) + np.nan
        pred_data = pd.DataFrame(init_mat, columns=col)  

        ## loop to simulate the responses in the block
        for t, row in block_data.iterrows():

            # predict stage: obtain input
            mag0     = row['mag0']
            mag1     = row['mag1']
            b_type   = row['b_type']
            state    = row['state']
            f_type   = row['feedback_type']
            if is_eval: act    = row['humanAct']

            mem      = {'b_type': b_type, 'f_type': f_type, 
                        'mag0': mag0, 'mag1': mag1}
            subj.buffer.push(mem)
            
            # control stage: make a resposne
            if is_eval:
                logAcc  = subj.control(state, mode='eval')
                logLike = subj.control(act,   mode='eval')     
            else:
                act, logAcc  = subj.control(state, rng=rng, mode='sample')

            match = (act==state)
            rew = ([mag0, mag1][state]) * match

            # record the vals 
            pred_data.loc[t, 'rew']     = rew 
            pred_data.loc[t, 'acc']     = np.exp(logAcc).round(3)
            pred_data.loc[t, 'act']     = act
            if is_eval: pred_data.loc[t, 'logLike'] = -logLike.round(3)

            for var in self.agent.voi:
                pred_data.loc[t, f'{var}'] = eval(f'subj.print_{var}()')

            # feedback stage: update the model 
            mem = {'b_type': b_type, 'f_type': f_type, 'state': state, 'act': act}
            subj.buffer.push(mem)  
            subj.learn() 

        # remove all nan columns
        pred_data = pred_data.dropna(axis=1, how='all')

        return pd.concat([block_data, pred_data], axis=1)