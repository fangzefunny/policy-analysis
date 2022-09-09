import numpy as np
from scipy.special import softmax
from scipy.stats import norm, gamma, beta

# get the machine epsilon
eps_ = 1e-12
max_ = 1e+12

sigmoid = lambda x: 1 / (1+np.exp(-x))

# ---------  Replay Buffer ----------- #

class simpleBuffer:
    '''Simple Buffer 2.0
    Update log: 
        To prevent naive writing mistakes,
        we turn the list storage into dict.
    '''
    def __init__(self):
        self.m = {}
        
    def push(self, m_dict):
        self.m = { k: m_dict[k] for k in m_dict.keys()}
        
    def sample(self, *args):
        lst = [self.m[k] for k in args]
        if len(lst)==1: return lst[0]
        else: return lst

# ---------  Model base -------------- #

class baseAgent:
    '''Base Agent'''
    name     = 'base'
    n_params = 0
    bnds     = []
    pbnds    = []
    p_name   = []  
    n_params = 0 
    p_priors = None 
    # value of interest, used for output
    # the interesting variable in simulation
    voi      = []
    
    def __init__(self, nA, params):
        self.nA = nA 
        self.load_params(params)
        self._init_Believes()
        self._init_Buffer()

    def load_params(self, params): 
        return NotImplementedError

    def _init_Buffer(self):
        self.buffer = simpleBuffer()
    
    def _init_Believes(self):
        self._init_Critic()
        self._init_Actor()
        self._init_Dists()

    def _init_Critic(self): pass 

    def _init_Actor(self): pass 

    def _init_Dists(self):  pass

    def learn(self): 
        return NotImplementedError

    def _policy(self): 
        return NotImplementedError

    def control(self, a, rng=None, mode='sample'):
        '''control problem 
            mode: -'get': get an action, need a rng 
                  -'eval': evaluate the log like 
        '''
        p_A = self._policy() 
        if mode == 'eval': 
            return np.log(p_A[a]+eps_)
        elif mode == 'sample': 
            return rng.choice(range(self.nA), p=p_A), np.log(p_A[a]+eps_)

class gagRL(baseAgent):
    name     = 'Gagne RL'
    bnds     = [(0, 1), (0, 1), (0, 30)]
    pbnds    = [(0,.5), (0,.5), (0, 10)]
    p_name   = ['α_STA', 'α_VOL', 'β']  
    n_params = len(bnds)
    voi      = ['ps', 'pi'] 
   
    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.beta      = params[2]

    def _init_Critic(self):
        self.p     = 1/2
        self.p_S   = np.array([1-self.p, self.p]) 

    def learn(self):
        self._learnCritic()

    def _learnCritic(self):
        c, o = self.buffer.sample('ctxt','state')
        alpha = self.alpha_sta if c=='sta' else self.alpha_vol
        self.p += alpha * (o - self.p)
        self.p_S = np.array([1-self.p, self.p])

    def _policy(self):
        m1, m2 = self.buffer.sample('mag0','mag1')
        mag = np.array([m1, m2])
        return softmax(self.beta * self.p_S * mag)

    def print_ps(self):
        return self.p

    def print_pi(self):
        return self._policy()[1]


# ---------  Two baselines ----------- #

class GagModel(gagRL):
    name     = 'Gagne best model'
    bnds     = [(0, 1), (0, 1), (0, 50), (0, 50), 
                (0, 1), (0, 50), (0, 1), (0, 1), (0, 1)]
    pbnds    = [(0,.5), (0,.5), (0, 10), (0, 10), 
                (0, 1), (0, 10), (0, 1), (0, 1), (0, 1)]
    p_name   = ['α_STA', 'α_VOL', 'β_STA', 'β_VOL', 
                'α_ACT', 'β_ACT', 'λ_STA', 'λ_ACT', 'r']  
    p_priors = [beta(a=2, b=2), beta(a=2, b=2), gamma(a=3, scale=3), gamma(a=3, scale=3),
                beta(a=2, b=2), gamma(a=3, scale=3), beta(a=2, b=2), beta(a=2, b=2), beta(a=2, b=2)]
    n_params = len(bnds)
    voi      = ['ps', 'pi'] 
   
    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.beta_sta  = params[2]
        self.beta_vol  = params[3]
        self.alpha_act = params[4]
        self.beta_act  = params[5]
        self.lamb_sta  = params[6]
        self.lamb_vol  = params[7]
        self.r         = params[8]
    
    def learn(self):
        self._learnCritic()
        self._learnActor()

    def _init_Actor(self):
        self.q     = 1/2
       
    def _learnActor(self):
        a = self.buffer.sample('act')
        self.q += self.alpha_act * (a - self.q)
       
    def _policy(self):
        c, m0, m1 = self.buffer.sample('ctxt', 'mag0','mag1')
        lamb = eval(f'self.lamb_{c}')
        v    = lamb*(self.p - (1-self.p)) \
               + (1-lamb)*abs(m1-m0)**self.r*np.sign(m1-m0)
        va   = eval(f'self.beta_{c}')*v + self.beta_act*(self.q - (1-self.q))
        pa   = 1 / (1 + np.exp(-va))
        return np.array([1-pa, pa])

class RlRisk(gagRL):
    name     = 'RL with risk preference'
    bnds     = [(0, 1), (0, 1), (0, 30), (0, 20), (0, 20)]
    pbnds    = [(0,.5), (0,.5), (0, 10), (0, 20), (0, 20)]
    p_name   = ['α_STA', 'α_VOL', 'β', 'γ_STA', 'γ_VOL']  
    n_params = len(bnds)
    p_priors = [beta(a=2, b=2), beta(a=2, b=2), gamma(a=3, scale=3), gamma(a=3, scale=3), gamma(a=3, scale=3)]
    voi      = ['ps', 'pi'] 
   
    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.beta      = params[2]
        self.gamma_sta = params[3]
        self.gamma_vol = params[4]
    
    def _learnCritic(self):
        c, o = self.buffer.sample('ctxt','state')
        self.p += eval(f'self.alpha_{c}') * (o - self.p)
        ps = np.clip(eval(f'self.gamma_{c}')*(self.p-.5)+.5, 0, 1)
        self.p_S = np.array([1-ps, ps])


# ---------  Mixture models ---------- #

class MixPol(baseAgent):
    name     = 'mixture policy model'
    bnds     = [(0,50), (0,50), (0,50), (0,50),
                (-40,40), (-40,40), (-40,40),
                (-40,40), (-40,40), (-40,40)]
    pbnds    = [(0, 2), (0, 2), (0, 3), (0, 5),
                (-5, 5), (-5, 5), (-5, 5),
                (-5, 5), (-5, 5), (-5, 5),]
    p_name   = ['α_STA', 'α_VOL', 'α_ACT', 'β',
                'λ0_STA', 'λ1_STA', 'λ2_STA',
                'λ0_VOL', 'λ1_VOL', 'λ2_VOL']
    p_priors = [gamma(a=3, scale=3), gamma(a=3, scale=3), gamma(a=3, scale=3), gamma(a=3, scale=3),
                norm(loc=0, scale=10), norm(loc=0, scale=10), norm(loc=0, scale=10), norm(loc=0, scale=10),
                norm(loc=0, scale=10), norm(loc=0, scale=10), norm(loc=0, scale=10), norm(loc=0, scale=10)]
    n_params = len(bnds)
    voi      = ['ps', 'pi', 'alpha', 'w1', 'w2', 'w3', 'l1', 'l2', 'l3']

    def load_params(self, params):
        self.alpha_sta = params[0]
        self.alpha_vol = params[1]
        self.alpha_act = params[2]
        self.beta      = params[3]
        self.l0_sta    = params[4]
        self.l1_sta    = params[5]
        self.l2_sta    = params[6]
        self.l0_vol    = params[7]
        self.l1_vol    = params[8]
        self.l2_vol    = params[9]
    
    def _init_Critic(self):
        self.theta = 0 
        self.p     = sigmoid(self.theta)
        self.p_S   = np.array([1-self.p, self.p]) 
    
    def _init_Actor(self):
        self.phi = 0 
        self.q   = sigmoid(self.phi)
        self.q_A = np.array([1-self.q, self.q]) 

    #  ------ learning the probability ------- #

    def learn(self):
        self._learnCritic()
        self._learnActor()

    def _learnCritic(self):
        c, o = self.buffer.sample('ctxt','state')
        alpha = self.alpha_sta if c=='sta' else self.alpha_vol
        self.theta += alpha * (o - self.p)
        self.p = sigmoid(self.theta)
        self.p_S = np.array([1-self.p, self.p])

    def _learnActor(self):
        a = self.buffer.sample('act')
        self.phi += self.alpha_act * (a - self.q)
        self.q = sigmoid(self.phi)
        self.q_A = np.array([1-self.q, self.q])

    #  ------ response strategy ------- #

    def _policy(self):
        c, m0, m1 = self.buffer.sample('ctxt', 'mag0','mag1')
        mag = np.array([m0, m1])
        pi_SM = softmax(self.beta*self.p_S*mag)
        pi_M  = softmax(self.beta*mag)
        w0, w1, w2 = self.get_w(c)
        # creat the mixature model 
        return w0*pi_SM + w1*pi_M + w2*self.q_A 

    def get_w(self, c):
        l0 = eval(f'self.l0_{c}')
        l1 = eval(f'self.l1_{c}')
        l2 = eval(f'self.l2_{c}')
        return softmax([l0, l1, l2])

    #  ------ print variable of interests ------- #

    def print_ps(self):
        return self.p

    def print_pi(self):
        return self._policy()[1]

    def print_alpha(self):
        return eval(f'self.alpha_{self.buffer.sample("ctxt")}') 

    def print_w1(self):
        return self.get_w(self.buffer.sample("ctxt"))[0]

    def print_w2(self):
        return self.get_w(self.buffer.sample("ctxt"))[1] 

    def print_w3(self):
        return self.get_w(self.buffer.sample("ctxt"))[2]  

    def print_l1(self):
        return eval(f'self.l0_{self.buffer.sample("ctxt")}')

    def print_l2(self):
        return eval(f'self.l1_{self.buffer.sample("ctxt")}')

    def print_l3(self):
        return eval(f'self.l2_{self.buffer.sample("ctxt")}')