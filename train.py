import pandas as pd 
import numpy as np 

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

import category_encoders as ce 

import sys
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss

from itertools import product
import pickle
from sklearn.preprocessing import StandardScaler
import time 
assert len(sys.argv) == 2, "Please give a config file!"
conf = sys.argv[1]
import_string = "from configs.%s import *" % sys.argv[1]
exec(import_string)

class Encoder(object):
	"""docstring for Encoder"""
	def __init__(self, feature_encoder = []):
		# feature_encoder [encodertype,[features],{params})] 
		super(Encoder, self).__init__()
		self.feature_encoder = feature_encoder
		#print(self.feature_encoder)
		self.encoders = {}

	def fit(self, X,y):
		for i, val in enumerate(self.feature_encoder):
			key,features, params = val[0], val[1], val[2]
			#print(params)
			#print(features)
			#print(X)
			#print(key)

			if key == 'OrdinalEncoder':
				#print(params)
				new_params = []
				for val in params['mapping']:
					#print(val)
					#print(features)
					if val['col'] in features:
						new_params.append(val)
				new_root_params = {'mapping': new_params}
				encoder = vars(ce)[key](return_df=True, **new_root_params)
				#print(new_params)
			else:
				encoder = vars(ce)[key](return_df=True, **params)
			
			if key == "TargetEncoder":
				encoder.fit(X[features], y)
			else:
				encoder.fit(X[features], y)

			#print(i)
			self.encoders[i] = encoder 

	def transform(self, X,y):
		data = []
		for key in self.encoders:
			#print(key)
			#print(X[self.feature_encoder[key]])

			d_encoded = self.encoders[key].transform(X[self.feature_encoder[key][1]])
			#print(d_encoded)
			data.append(d_encoded.values)
		data = np.concatenate(data,axis=1)
		return data 



def analysis_results(cv_result):
	# dump to disk
	pass

def train(estimator,encoder, X, y, seed):
	n_cv = 5
	kf = KFold(n_splits = n_cv )
	scores = []

	encode_time, train_time = 0, 0
	for train_index, test_index in kf.split(X):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		t = time.time()
		

		encoder.fit(X_train, y_train)
		X_train = encoder.transform(X_train, y_train)
		X_test  = encoder.transform(X_test,y_test)
		encode_time += time.time() - t
		ss = StandardScaler()
		print(X_train.shape)
		ss.fit(X_train)
		X_train, X_test = ss.transform(X_train), ss.transform(X_test)
		t = time.time()
		estimator.fit(X_train, y_train)
		y_train_pred = estimator.predict_proba(X_train)[:,1]
		y_test_pred = estimator.predict_proba(X_test)[:,1]

		# try:
		# 	y_train_pred = estimator.predict_proba(X_train, y_train)
		# 	y_test_pred = estimator.predict_proba(X_test, y_test)

		# except:
		# 	#TODO
		# 	pass
		train_time += time.time() - t

		train_auc = roc_auc_score(y_train, y_train_pred)
		test_auc = roc_auc_score(y_test, y_test_pred)
		train_logloss = log_loss(y_train, y_train_pred)
		test_logloss = log_loss(y_test, y_test_pred)
		scores.append(((train_auc, train_logloss),(test_auc, test_logloss)))
	
	return scores, encode_time/5, train_time/5



def train_search(abs_estimator, params,  X, y, seed,ordinalmap, name='def', idx=0):
	
	#TODO our combinations of encoder vs feature type, encoder vs feature size
	#hash n_components=6
	
	encoder_deep_params  = {'OrdinalEncoder':{'mapping':ordinalmap},'BinaryEncoder':{},'HashingEncoder': {'n_components':16},'OneHotEncoder':{},'TargetEncoder':{}}
	encoder_names = ['OrdinalEncoder','BinaryEncoder', 'HashingEncoder','TargetEncoder','OneHotEncoder']
	feature_groups_1 = [['bin_%d' % i for i in range(5)], 
		['nom_%d' % i for i in range(10)], 
		['ord_%d' % i for i in range(6)] + ['day','month'] ]
	feature_groups_2 = [['bin_%d' % i for i in range(5)] + ['nom_%d' % i for i in range(5)] + ['ord_%d' % i for i in range(5)] + ['day','month'],
		['nom_7', 'nom_8', 'ord_5'], ['nom_5','nom_6','nom_9']]
	
	feature_groups_3 = [['bin_%d' % i for i in range(5)] + ['nom_%d' % i for i in range(5)] + ['ord_%d' % i for i in range(5)] + ['day','month'],
		['nom_7', 'nom_8', 'ord_5'] + ['nom_5','nom_6','nom_9']]
	
	#encoders_params_combinations = [[('OneHotEncoder', feature_groups_3[0], {}), (encoder_names[i], feature_groups_3[1], encoder_deep_params[encoder_names[i]])] for i in range(len(encoder_names)) ] 

	#print(encoders_params_combinations)

	# encoders_params_combinations = list(product([[encoder_names[i], feature_groups_1[0], encoder_deep_params[encoder_names[i]]] for i in range(len(encoder_names))],
	# [[encoder_names[i], feature_groups_1[1], encoder_deep_params[encoder_names[i]]] for i in range(len(encoder_names))],
	# [[encoder_names[i], feature_groups_1[2], encoder_deep_params[encoder_names[i]]] for i in range(len(encoder_names))])) + 

	encoders_params_combinations = list(product([[encoder_names[i], feature_groups_2[0], encoder_deep_params[encoder_names[i]]] for i in range(len(encoder_names))],
	[[encoder_names[i], feature_groups_2[1], encoder_deep_params[encoder_names[i]]] for i in range(len(encoder_names))],
	[[encoder_names[i], feature_groups_2[2], encoder_deep_params[encoder_names[i]]] for i in range(len(encoder_names))]))

	#print(encoders_params_combinations[0])

	scores, encode_times, train_times = [], [], []

	for encoder_params in tqdm(encoders_params_combinations):
	
		encoder = Encoder(feature_encoder=encoder_params)
		estimator = abs_estimator( **params)	
		score, encode_time, train_time = train(estimator,encoder, X, y,seed)
		scores.append(score)
		encode_times.append(encode_time)
		train_times.append(train_time)

		print(np.array([score[i][0][0] for i in range(len(score))]).mean(),
			np.array([score[i][1][0] for i in range(len(score))]).mean(), train_time, encode_time)

	f = open('out/%s_%d_res.pkl' % (name, idx), 'wb')
	pickle.dump([scores,encode_times, train_times, params],f)
	f.close()


if __name__ == "__main__":
	# Parameters from a config file
	#############################
	# abs_estimator
	# params
	# seed
	# n_search
	# ###########################
	ord_3 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm','n', 'o', -1]
	ord_4 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M','N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',-1]
	ord_5 = ['AG', 'AI', 'AU', 'AW', 'Ay', 'BL', 'BX', 'Bx', 'CN', 'CU', 'Cn',
			'DI', 'DN', 'DR', 'DT', 'Dj', 'Dn', 'EC', 'Ey', 'FB', 'FH', 'Fl',
			'GZ', 'HF', 'HK', 'HO', 'Hk', 'IA', 'IS', 'Ib', 'In', 'Io', 'Iq',
			'JQ', 'JT', 'Ji', 'Kq', 'LS', 'LY', 'Lo', 'MF', 'MU', 'MV', 'MX',
			'Mg', 'Mq', 'NS', 'NT', 'Nh', 'OM', 'OZ', 'Oe', 'Ox', 'PG', 'PS',
			'Pk', 'Pw', 'QV', 'Qm', 'RB', 'RD', 'RT', 'RV', 'Re', 'Rj', 'Ro',
			'Rv', 'Rz', 'SL', 'SS', 'Sk', 'Sz', 'TH', 'TL', 'TP', 'TZ', 'Tg',
			'Ty', 'Tz', 'US', 'UV', 'WC', 'WW', 'Wr', 'XC', 'XI', 'XM', 'XR',
			'XU', 'YJ', 'Yb', 'Yi', 'Yr', 'Zv', 'aA', 'aE', 'al', 'be', 'cR',
			'cY', 'cg', 'cy', 'dh', 'dp', 'eA', 'eN', 'ep', 'fF', 'fO', 'fV',
			'fe', 'gK', 'gL', 'gV', 'gc', 'gj', 'gt', 'hG', 'hT', 'ht', 'hx',
			'iS', 'iv', 'ja', 'jf', 'jp', 'kB', 'kP', 'kT', 'kv', 'lA', 'lR',
			'lS', 'ly', 'mD', 'mP', 'mX', 'mi', 'mo', 'nS', 'ne', 'nf', 'nj',
			'nn', 'oI', 'oJ', 'oU', 'oh', 'ok', 'pB', 'pE', 'pT', 'pZ', 'pl',
			'qN', 'qP', 'rA', 'rM', 'rg', 'rl', 'sF', 'sY', 'sc', 'sf', 'tT',
			'th', 'tn', 'uI', 'uP', 'uQ', 'uW', 'uZ', 'ur', 'us', 'vQ', 'vq',
			'vw', 'vx', 'wJ', 'wU', 'wa', 'xB', 'xF', 'xG', 'yE', 'yK', 'zc',
			'ze', 'zf', 'zp',-1]

	data = pd.read_csv('data/train.csv')
	features = list(data.columns.values)
	features.remove('id')
	features.remove('target')
	data2 = data.dropna()
	ordinalmap = []
	for val in features:
		if val in ['ord_0','ord_1','ord_2','ord_3','ord_4','ord_5','day','month']:
			continue
		unique = data[val].unique()
		
		maps = {unique[i]:i for i in range(len(unique))}
		maps[-1] = len(unique) 
		print(maps)
		ordinalmap.append({'col':val, 'mapping':maps})
	ordinalmap.extend([{'col':'ord_0','mapping':{1:0,2:1, 3:2, -1:3} },
	{'col':'ord_1', 'mapping':{'Novice':0, 'Contributor':1, 'Expert':2,'Master':3,'Grandmaster':4,-1:5 }},
	{'col':'ord_2', 'mapping':{'Freezing':0, 'Cold':1, 'Warm':2,'Hot':3,'Boiling Hot':4,'Lava Hot':5, -1:6 }},
	{'col':'ord_3', 'mapping':{ord_3[i]:i for i in range(len(ord_3))}},
	{'col':'ord_4', 'mapping':{ord_4[i]:i for i in range(len(ord_4))}},
	{'col':'ord_5', 'mapping':{ord_5[i]:i for i in range(len(ord_5))}}])



	data = data.fillna(-1)
	
	data = data.sample(5000,random_state=seed ).reindex()

	
	X, y = data[features], data['target']

	estimator_params = {}
	for i in range(20):

		for key in params:
			if hasattr(params[key], '__iter__'):
				np.random.seed(seed+i)
				estimator_params[key] = params[key][np.random.randint(0,len(params[key]))]
			elif callable(params[key]):
				estimator_params[key] = params[key](seed=seed+i)
			else:
				estimator_params[key] = params[key]
		#print(estimator_params)
		train_search(abs_estimator, estimator_params, X, y, seed,ordinalmap, name=conf, idx=i)

