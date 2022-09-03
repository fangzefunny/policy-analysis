

models = {
    'RF': RandomForestClassifier,
    'GB': GradientBoostingClassifier,
}
res = {k: [] for k in models.keys()}

for train_index, test_index in skf.split(x, y):

    #
    #
    #
    #
    

    for m in models:
        tem_res = {}
        
        clf = models[m]()
