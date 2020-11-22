from sklearn.linear_model import LogisticRegression

seed = 0
estimator = LogisticRegression( random_state=seed)
distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])

