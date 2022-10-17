import numpy as np
from scipy.special import softmax
from scipy.stats import norm, gamma, beta

# get the machine epsilon
eps_ = 1e-12
max_ = 1e+12

# ---------  Some Functions ----------- #

sigmoid = lambda x: 1 / (1+np.exp(-x))

flatten = lambda l: [item for sublist in l for item in sublist]

def get_param_name(params, block_types=['sta', 'vol'], feedback_types=['gain', 'loss']):
    return flatten([flatten([[f'{key}_{j}_{i}' for key in params]
                                               for i in feedback_types])
                                               for j in block_types])
                    

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
    bnds     = [(0, 30)] + [(0, 1)]*4
    pbnds    = [(0, 10)] + [(0,.5)]*4
    p_name   = ['β'] + get_param_name(['α'])
    n_params = len(bnds)
    voi      = ['ps', 'pi'] 
   
    def load_params(self, params):
        self.beta           = params[0]
        self.alpha_sta_gain = params[1]
        self.alpha_sta_loss = params[2]
        self.alpha_vol_gain = params[3]
        self.alpha_vol_loss = params[4]
        
    def _init_Critic(self):
        self.p     = 1/2
        self.p_S   = np.array([1-self.p, self.p]) 

    def learn(self):
        self._learnCritic()

    def _learnCritic(self):
        b, f, o = self.buffer.sample('b_type', 'f_type', 'state')
        self.p += eval(f'self.alpha_{b}_{f}') * (o - self.p)
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

class FLR(gagRL):
    name     = 'flexible learning rate'
    bnds     = [(0, 1), (0, 50), (0, 1)] + [(0, 1), (0, 50), (0, 1)]*4
    pbnds    = [(0,.5), (0, 10), (0, 1)] + [(0,.5), (0, 10), (0, 1)]*4
    p_name   = ['α_act', 'β_act', 'r'] + get_param_name(['α', 'β', 'λ'])
    p_priors = [beta(a=2, b=2), gamma(a=3, scale=3), gamma(a=3, scale=3)] + \
                [beta(a=2, b=2), gamma(a=3, scale=3), beta(a=2, b=2)]*4
    n_params = len(bnds)
    voi      = ['ps', 'pi', 'alpha'] 
   
    def load_params(self, params):

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.beta_act       = params[1]
        self.r              = params[2]

        # ---- Stable & gain ---- #
        self.alpha_sta_gain = params[3]
        self.beta_sta_gain  = params[4]
        self.lamb_sta_gain  = params[5]

        # ---- Stable & loss ---- #
        self.alpha_sta_loss = params[6]
        self.beta_sta_loss  = params[7]
        self.lamb_sta_loss  = params[8]

        # ---- Volatile & gain ---- #
        self.alpha_vol_gain = params[9]
        self.beta_vol_gain  = params[10]
        self.lamb_vol_gain  = params[11]

        # ---- Voatile & gain ---- #
        self.alpha_vol_loss = params[12]
        self.beta_vol_loss  = params[13]
        self.lamb_vol_loss  = params[14]
        
    
    def learn(self):
        self._learnCritic()
        self._learnActor()

    def _init_Actor(self):
        self.q     = 1/2
       
    def _learnActor(self):
        a = self.buffer.sample('act')
        self.q += self.alpha_act * (a - self.q)
       
    def _policy(self):
        b, f, m0, m1 = self.buffer.sample('b_type', 'f_type','mag0','mag1')
        lamb = eval(f'self.lamb_{b}_{f}')
        v    = lamb*(self.p - (1-self.p)) \
               + (1-lamb)*abs(m1-m0)**self.r*np.sign(m1-m0)
        va   = eval(f'self.beta_{b}_{f}')*v + self.beta_act*(self.q - (1-self.q))
        pa   = 1 / (1 + np.exp(-va))
        return np.array([1-pa, pa])
    
    def print_alpha(self):
        b, f = self.buffer.sample('b_type', 'f_type')
        return eval(f'self.alpha_{b}_{f}') 

class RP(gagRL):
    name     = 'risk preference'
    bnds     = [(0, 30)] + [(0, 1), (0, 20)]*4
    pbnds    = [(0, 10)] + [(0,.5), (0, 20)]*4
    p_name   = ['β'] + get_param_name(['α', 'γ'])
    n_params = len(bnds)
    p_priors = [gamma(a=3, scale=3)] + [beta(a=2, b=2), gamma(a=3, scale=3)]*4
    voi      = ['ps', 'pi'] 
   
    def load_params(self, params):

        # ---- General ----- #
        self.beta           = params[0]

        # ---- Stable & gain ---- #
        self.alpha_sta_gain = params[1]
        self.gamma_sta_gain = params[2]

        # ---- Stable & loss ---- #
        self.alpha_sta_loss = params[3]
        self.gamma_sta_loss = params[4]

        # ---- Volatile & gain ---- #
        self.alpha_vol_gain = params[5]
        self.gamma_vol_gain = params[6]

        # ---- Voatile & gain ---- #
        self.alpha_vol_loss = params[7]
        self.gamma_vol_loss = params[8]
       
    def _learnCritic(self):
        b, f, o = self.buffer.sample('b_type', 'f_type', 'state')
        self.p += eval(f'self.alpha_{b}_{f}') * (o - self.p)
        ps = np.clip(eval(f'self.gamma_{b}_{f}')*(self.p-.5)+.5, 0, 1)
        self.p_S = np.array([1-ps, ps])

class MOS(gagRL):
    name     = 'mixture of strategy'
    bnds     = [(0, 1), (0,50)] + ([(0, 1)]+[(-40,40)]*3) * 4
    pbnds    = [(0,.5), (0, 5)] + ([(0,.5)]+[(-5, 5)]*3) * 4
    p_name   = ['α_act', 'β']   + get_param_name(['α', 'λ1', 'λ2', 'λ3'])
    p_priors = [beta(a=2, b=2), gamma(a=3, scale=3)] + \
                    ([beta(a=2, b=2)]+[norm(loc=0, scale=10)]*3) * 4
    n_params = len(bnds)
    voi      = ['ps', 'pi', 'alpha', 'w1', 'w2', 'w3', 'l1', 
                'l2', 'l3', 'l1_effect', 'l2_effect', 'l3_effect']

    def load_params(self, params):

        # ---- General ----- #
        self.alpha_act      = params[0]
        self.beta           = params[1]

        # ---- Stable & gain ---- #
        self.alpha_sta_gain = params[2]
        self.l0_sta_gain    = params[3]
        self.l1_sta_gain    = params[4]
        self.l2_sta_gain    = params[5]

        # ---- Stable & loss ---- #
        self.alpha_sta_loss = params[6]
        self.l0_sta_loss    = params[7]
        self.l1_sta_loss    = params[8]
        self.l2_sta_loss    = params[9]

        # ---- Volatile & gain ---- #
        self.alpha_vol_gain = params[10]
        self.l0_vol_gain    = params[11]
        self.l1_vol_gain    = params[12]
        self.l2_vol_gain    = params[13]

        # ---- Volatile & loss ---- #
        self.alpha_vol_loss = params[14]
        self.l0_vol_loss    = params[15]
        self.l1_vol_loss    = params[16]
        self.l2_vol_loss    = params[17]
    
    def _init_Critic(self):
        self.p   = .5
        self.p_S = np.array([1-self.p, self.p]) 
    
    def _init_Actor(self):
        self.q   = .5
        self.q_A = np.array([1-self.q, self.q]) 
        self.pi_effect = [1/3, 1/3, 1/3]

    #  ------ learning the probability ------- #

    def learn(self):
        self._learnCritic()
        self._learnActor()

    def _learnCritic(self):
        b, f, o = self.buffer.sample('b_type', 'f_type', 'state')
        self.p += eval(f'self.alpha_{b}_{f}') * (o - self.p)
        self.p_S = np.array([1-self.p, self.p])

    def _learnActor(self):
        a = self.buffer.sample('act')
        self.q += self.alpha_act * (a - self.q)
        self.q_A = np.array([1-self.q, self.q])

    #  ------ response strategy ------- #

    def _policy(self):
        b, f, m0, m1 = self.buffer.sample('b_type', 'f_type', 'mag0','mag1')
        mag = np.array([m0, m1])
        pi_SM = softmax(self.beta*self.p_S*mag)
        pi_M  = softmax(self.beta*mag)
        w0, w1, w2 = self.get_w(b, f)
        # creat the mixature model 
        self.pi_effect = [pi_SM[1], pi_M[1], self.q_A[1]]
        return w0*pi_SM + w1*pi_M + w2*self.q_A 

    def get_w(self, b, f):
        l0 = eval(f'self.l0_{b}_{f}')
        l1 = eval(f'self.l1_{b}_{f}')
        l2 = eval(f'self.l2_{b}_{f}')
        return softmax([l0, l1, l2])

    #  ------ print variable of interests ------- #

    def print_ps(self):
        return self.p

    def print_pi(self):
        return self._policy()[1]

    def print_alpha(self):
        b, f = self.buffer.sample('b_type', 'f_type')
        return eval(f'self.alpha_{b}_{f}') 

    def print_w1(self):
        b, f = self.buffer.sample('b_type', 'f_type')
        return self.get_w(b, f)[0]

    def print_w2(self):
        b, f = self.buffer.sample('b_type', 'f_type')
        return self.get_w(b, f)[1]

    def print_w3(self):
        b, f = self.buffer.sample('b_type', 'f_type')
        return self.get_w(b, f)[2]  

    def print_l1(self):
        b, f = self.buffer.sample('b_type', 'f_type')
        return eval(f'self.l0_{b}_{f}')

    def print_l2(self):
        b, f = self.buffer.sample('b_type', 'f_type')
        return eval(f'self.l1_{b}_{f}')

    def print_l3(self):
        b, f = self.buffer.sample('b_type', 'f_type')
        return eval(f'self.l2_{b}_{f}')

    def print_l1_effect(self):
        return self.pi_effect[0]

    def print_l2_effect(self):
        return self.pi_effect[1]

    def print_l3_effect(self):
        return self.pi_effect[2]

# ---------  Mixture models ---------- #

class MixPol(MOS):
    name     = 'mixture policy model'
    bnds     = [(0,50), (0,50)] + ([(0,50)]+[(-40,40)]*3) * 4
    pbnds    = [(0, 3), (0, 5)] + ([(0, 2)]+[(-5, 5)]*3) * 4
    p_name   = ['α_act', 'β']   + get_param_name(['α', 'λ1', 'λ2', 'λ3'])
    p_priors = [gamma(a=3, scale=3), gamma(a=3, scale=3)] + \
                    ([gamma(a=3, scale=3)]+[norm(loc=0, scale=10)]*3) * 4
    n_params = len(bnds)
    voi      = ['ps', 'pi', 'alpha', 'w1', 'w2', 'w3', 'l1', 
                'l2', 'l3', 'l1_effect', 'l2_effect', 'l3_effect']

    def _init_Critic(self):
        self.theta = 0 
        self.p     = sigmoid(self.theta)
        self.p_S   = np.array([1-self.p, self.p]) 
    
    def _init_Actor(self):
        self.phi = 0 
        self.q   = sigmoid(self.phi)
        self.q_A = np.array([1-self.q, self.q]) 
        self.pi_effect = [1/3, 1/3, 1/3]

    #  ------ learning the probability ------- #

    def learn(self):
        self._learnCritic()
        self._learnActor()

    def _learnCritic(self):
        b, f, o = self.buffer.sample('b_type', 'f_type', 'state')
        self.theta += eval(f'self.alpha_{b}_{f}') * (o - self.p)
        self.p = sigmoid(self.theta)
        self.p_S = np.array([1-self.p, self.p])

    def _learnActor(self):
        a = self.buffer.sample('act')
        self.phi += self.alpha_act * (a - self.q)
        self.q = sigmoid(self.phi)
        self.q_A = np.array([1-self.q, self.q])
