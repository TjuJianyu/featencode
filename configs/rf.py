
from sklearn.ensemble import RandomForestClassifier
import numpy as np
seed = 0
n_jobs=32
abs_estimator = RandomForestClassifier
params = dict(max_depth=[1,2,3,4,5], min_samples_split=[2,8,16,32,64,128],n_jobs=n_jobs)
n_search = 20
