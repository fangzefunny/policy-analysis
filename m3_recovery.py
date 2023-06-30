import argparse 
import os 
import pickle
import datetime 
import numpy as np
import pandas as pd

from utils.parallel import get_pool 
from utils.model import *
from utils.env_fn import *

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--n_fit',      '-f', help='fit times', type = int, default=1)
parser.add_argument('--data_set',   '-d', help='which_data', type = str, default='exp1data')
parser.add_argument('--env_name',   '-e', help='which environment', type = str, default='rl_reversal')
parser.add_argument('--method',     '-m', help='methods, mle or map', type = str, default='hier')
parser.add_argument('--agent_name', '-n', help='choose agent', default='MOS6')
parser.add_argument('--n_cores',    '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=1)
parser.add_argument('--seed',       '-s', help='random seed', type=int, default=420)
args = parser.parse_args()
args.agent = eval(args.agent_name)
env =  eval(args.env_name)
args.group = 'group' if args.method=='hier' else 'ind'