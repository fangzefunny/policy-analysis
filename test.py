import pickle 


fname = f'fits/exp1data/fit_sub_info-MOS-bms.pkl'
with open(fname, 'rb')as handle: 
    fit_sub_info = pickle.load(handle)

print(1)
    