import pandas as pd 
import numpy as np 

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import RandomizedSearchCV

import category_encoders as ce 

import sys


assert len(sys.argv) == 2, "Please give a config file!"
conf = sys.argv[1]
import_string = "from configs.%s import *" % sys.argv[1]
exec(import_string)

class Encoder(object):
	"""docstring for Encoder"""
	def __init__(self, etype='label',feature_encoder = {}):
		super(Encoder, self).__init__()
		self.feature_encoder = feature_encoder
	
	def fit_transform(X,y):
		for key in self.feature_encoder:
			etype = feature_encoder[key]
			if etype == 'label':
				pass
			elif etype == 'target':
				pass
			elif etype == 'binary':
				pass
			elif etype == 'hash':
				pass
			elif etype == 'onehot':
				pass
			elif etype == 'effect':
				pass
		return (encoded_X, y)



def analysis_results(cv_result):
	# dump to disk
	pass

def train(estimator,encoder,params,X, y, seed, n_jobs=1):

	clf = RandomizedSearchCV(estimator, params, 
		scoring='auc',
		cv = 5, n_jobs = n_jobs,
		return_train_score = True, random_state=seed)
	pipe = Pipeline([('scaler', encoder), clf])
	search = pipe.fit(X,y)
	analysis_results(search.cv_results_)



def train_search(estimator, params, seed, n_jobs=1):
	
	#TODO our combinations of encoder vs feature type, encoder vs feature size
	encoders_combinations = []

	for encoder in encoders_combinations:
		train(estimator,encoder,params, X, y,seed,n_jobs)

if __name__ == "__main__":
	# Parameters from a config file
	#############################
	# estimator
	# params
	# seed
	# ###########################
	train_search(estimator, params, seed)








