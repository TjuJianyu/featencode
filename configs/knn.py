from sklearn.neighbors import KNeighborsClassifier
seed = 0
n_jobs=32
abs_estimator = KNeighborsClassifier
params = dict(n_neighbors=[3,5,7,10,15,20,30,40],n_jobs=n_jobs, weights=['uniform','distance'], p=[1,2])
n_search = 20
