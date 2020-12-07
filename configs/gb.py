from sklearn.ensemble import GradientBoostingClassifier
seed=0
abs_estimator = GradientBoostingClassifier
params = dict(max_depth=[1,2,3,4,5], min_samples_split=[2,8,16,32,64,128])
n_search = 20
