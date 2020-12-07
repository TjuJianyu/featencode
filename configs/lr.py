from sklearn.linear_model import LogisticRegression
import numpy as np
def func(seed):
	np.random.seed(seed)
	return np.random.rand()*4

seed = 0
abs_estimator = LogisticRegression
params = dict(C=func, penalty=['l2'],max_iter=1000, solver=['lbfgs'])
n_search = 20
