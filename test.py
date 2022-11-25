import pickle 
from utils.bms import fit_bms

models = ['MOS', 'FLR']
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

bms_results = fit_bms(fit_sub_info, use_bic=True)


