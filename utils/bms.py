import numpy as np 
import warnings

from scipy.special import softmax, psi, gammaln
from scipy.stats import gamma

eps_ = 1e-16

def fit_bms(fit_sub_info, use_bic=False, tol=1e-4):
    '''Fit group-level Bayesian model seletion
    Nm is the number of model
    Args: 
        fit_sub_info: [Nm, list] a list of model fitting results
        use_bic: use bic to approximate lme
        tol: 

    Outputs:
        BMS result: a dict including 
            -alpha: [1, Nm] posterior of the model probability
            -p_m1D: [nSub, Nm] posterior of the model 
                     assigned to the subject data p(m|D)
            -E_r1D: [nSub, Nm] expectation of E[p(r|D)]
            -xp:    [Nm,] exceedance probabilities
            -bor:   [1] Bayesian Omnibus Risk, the probability
                    of choosing null hypothesis: model frequencies are equal
            -pxp:   [Nm,] protected exceedance probabilities
    ----------------------------------------------------------------
    REFERENCES:
    
    Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ (2009)
    Bayesian Model Selection for Group Studies. NeuroImage 46:1004-1017
    
    Rigoux, L, Stephan, KE, Friston, KJ and Daunizeau, J. (2014)
    Bayesian model selection for group studiesRevisited.
    NeuroImage 84:971-85. doi: 10.1016/j.neuroimage.2013.08.065
    -------------------------------------------------------------------
    Based on: https://github.com/sjgershm/mfit
    @ ZF
    '''
    ## get log model evidence
    if use_bic:
        lme = np.vstack([-.5*np.array(fit_info['bic']) for fit_info in fit_sub_info]).T
    else:
        lme = np.vstack([calc_lme(fit_info) for fit_info in fit_sub_info]).T
    
    ## get group-level posterior
    Nm = lme.shape[1]
    alpha0, alpha = np.ones([1, Nm]), np.ones([1, Nm])

    while True:
        
        # cache previous α
        prev = alpha.copy()

        # compute the posterior: Nsub x Nm
        # p(m|D) (p, k) = exp[log p(D(p,1))|m(p,k)) + Psi(α(1,k)) - Psi(α'(1,1))]
        log_u = lme + psi(alpha) - psi(alpha.sum())
        u = np.exp(log_u - log_u.max(1, keepdims=True)) # the max trick 
        p_m1D = u / u.sum(1, keepdims=True)

        # compute beta: 1 x Nm
        # β(k) = sum_p p(m|D)
        B = p_m1D.sum(0, keepdims=True)

        # update alpha: 1 x Nm
        # α(k) = α0(k) + β(k) 
        alpha = alpha0 + B 

        # check convergence 
        if np.linalg.norm(alpha - prev) < tol:
            break 
    
    # get the expected posterior 
    E_r1D = alpha / alpha.sum()

    # get the exeedence probabilities 
    xp = dirchlet_exceedence(alpha)

    # get the Bayesian Omnibus risk
    bor = calc_BOR(lme, p_m1D, alpha, alpha0)
    
    # get the protected exeedence probabilities
    pxp=(1-bor)*xp+bor/Nm

    # out BMS fit 
    BMS_result = { 'alpha_post': alpha, 'p_m1D': p_m1D, 
                   'E_r1D': E_r1D, 'xp': xp, 'bor': bor, 'pxp': pxp}

    return BMS_result

def calc_lme(fit_info):
    '''Calculate Log Model Evidence
    Turn a list of fitting results of different
    model into a matirx lme. Ns means number of subjects, 
    Nm is the number of models.
    Args:
        fit_info: [dict,] A dict of model's fitting info
            - log_post: opt parameters
            - log_like: log likelihood
            - param: the optimal parameters
            - n_param: the number of parameters
            - aic
            - bic
            - H: hessian matrix 
    
    Outputs:
        lme: [Ns, Nm] log model evidence 
                
    '''
    lme  = []
    for s in range(len(fit_info['log_post'])):
        # log|-H|
        h = np.log(np.linalg.det(fit_info['H'][s]))
        # log p(D,θ*|m) + .5(log(d) - log|-H|) 
        l = fit_info['log_post'][s] + \
            .5*(fit_info['n_param']*np.log(2*np.pi)-h)
        lme.append(l)
        
    # use BIC if any Hessians are degenerate 
    ind = np.isnan(lme) | np.isinf(lme)| (np.imag(lme)!=0)
    if any(ind.reshape([-1])): 
        warnings.warn("Hessians are degenerated, use BIC")
        lme = -.5 * np.array(fit_info['bic'])
            
    return np.array(lme)

def dirchlet_exceedence(alpha_post, nSample=1e6):
    '''Sampling to calculate exceedence probability
    Args:
        alpha: [1,Nm] dirchilet distribution parameters
        nSample: number of samples
    Output: 
    '''
    # the number of categories
    Nm = alpha_post.shape[1]
    alpha_post = alpha_post.reshape([-1])

    # sampling in blocks
    blk = int(np.ceil(nSample*Nm*8 / 2**28))
    blk = np.floor(nSample/blk * np.ones([blk,]))
    blk[-1] = nSample - (blk[:-1]).sum()
    blk = blk.astype(int)

    # sampling 
    xp = np.zeros([Nm,])
    for i in range(len(blk)):

        # sample from a gamma distribution and normalized
        r = np.vstack([gamma(a).rvs(size=blk[i]) for a in alpha_post]).T

        # use the max decision rule and count 
        xp += (r == np.amax(r, axis=1, keepdims=True)).sum(axis=0)

    return xp / nSample

# -------- Bayesian Omnibus Risk -------- #

def calc_BOR(lme, p_m1D, alpha_post, alpha0):
    '''Calculate the Bayesian Omnibus Risk
     Args:
        lme: [Nsub, Nm] log model evidence
        p_r1D: [Nsub, Nm] the posterior of each model 
                        assigned to the data
        alpha_post:  [1, Nm] H1: alpha posterior 
        alpha0: [1, Nm] H0: alpha=[1,1,1...]
    Outputs:
        bor: the probability of selection the null
                hypothesis.
    '''
    # calculte F0 and F1
    f0 = F0(lme)
    f1 = FE(lme, p_m1D, alpha_post, alpha0)

    # BOR = 1/(1+exp(F1-F0))
    bor = 1 / (1+ np.exp(f1-f0))
    return bor 

def F0(lme):
    '''Calculate the negative free energy of H0
    Args:
        lme: [Nsub, Nm] log model evidence
    Outputs:
        f0: negative free energy as an approximation
            of log p(D|H0)
    '''
    Nm = lme.shape[1]
    qm = softmax(lme, axis=1)    
    f0 = (qm * (lme - np.log(Nm) - np.log(qm + eps_))).sum()                                  
    return f0
    
def FE(lme, p_m1D, alpha_post, alpha0):
    '''Calculate the negative free energy of H1
    Args:
        lme: [Nsub, Nm] log model evidence
        p_m1D: [Nsub, Nm] the posterior of each model 
                        assigned to the data
        alpha_post:  [1, Nm] H1: alpha posterior 
        alpha0: [1, Nm] H0: alpha=[1,1,1...]
    Outputs:
        f1: negative free energy as an approximation
            of log p(D|H1)
    '''
    E_log_r = psi(alpha_post) - psi(alpha_post.sum())
    E_log_rmD = (p_m1D*(lme+E_log_r)).sum() + ((alpha0 -1)*E_log_r).sum()\
                + gammaln(alpha0.sum()) - (gammaln(alpha0)).sum()
    Ent_p_m1D = -(p_m1D*np.log(p_m1D + eps_)).sum()
    Ent_alpha  = gammaln(alpha_post).sum() - gammaln(alpha_post.sum()) \
                                        - ((alpha_post-1)*E_log_r).sum()
    f1 = E_log_rmD + Ent_p_m1D + Ent_alpha
    return f1