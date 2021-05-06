from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


class Model:
	def __init__(self, clf, X, y):
		self.clf = clf
		self.X = X
		self.y = y
		self.train_X = self.train_y =\
		self.train_X_res = self.train_y_res =\
		self.test_X = self.test_y =\
		self.pred_y =\
		self.optimization =\
		self.f1_score = self.precision_score = self.recall_score = self.accuracy_score = None
	
	def str(self):
		str = ''
		str = str + f'Best params for classifier: {self.optimization.best_params_}\n'

		for metric in ['f1_score', 'precision_score', 'recall_score', 'accuracy_score']:
			str = str + f'{metric}:\t{getattr(self, metric)}\n'

		return str


class Builder:
	def __init__(self, clf, X, y): 
		self.model = Model(clf, X, y)
		self.hyperparameters_used = False
		self.class_balancing_used = False

	def build(self):
		return self.model


class TestTrainSplitBuilder(Builder):
	def test_train_split(self, split_size):
		self.model.train_X, self.model.test_X,\
		self.model.train_y, self.model.test_y = train_test_split(self.model.X, self.model.y, test_size = split_size, shuffle = False)
		return self
    

class ClassBalancingBuilder(TestTrainSplitBuilder):
	def SMOTE(self):
		print('Éxecuting SMOTE...')
		self.class_balancing_used = True
		sm = SMOTE(random_state=42)
		self.model.train_X_res, self.model.train_y_res = sm.fit_resample(self.model.train_X, self.model.train_y)
		return self
    
	def SMOTETomek(self):
		print('Executing SMOTETomek...')
		return self


class HyperparametersTuningBuilder(ClassBalancingBuilder):
	def tune_hyperparameters(self, cv_params):
		self.hyperparameters_used = True
		
		# TODO: take all params from function params
		self.model.optimization = GridSearchCV(\
			self.model.clf,\
			cv_params,\
			scoring = 'f1_macro', cv = 10, n_jobs = -1, verbose = True\
		)
		
		if self.class_balancing_used:
			print('Fitting optimization with resampled data...')
			self.model.optimization.fit(self.model.train_X_res, self.model.train_y_res)
		else:
			print('Fitting optimization...')
			self.model.optimization.fit(self.model.train_X, self.model.train_y)

		return self


class TrainModelBuilder(HyperparametersTuningBuilder):
	def train_model_with_optimization(self, use_resampled_data = False, use_n_best_features = 'all', use_only_important_features = False):
		if (not self.class_balancing_used and use_resampled_data):
			raise RuntimeError('Resampling was not executed. Resample data firstly.')
		
		data_to_use = [self.model.train_X_res, self.model.train_y_res] if use_resampled_data else [self.model.train_X, self.model.train_y]

		if (use_only_important_features):
			important_features = []
			for name, importance in zip(self.model.train_X.columns, self.model.optimization.best_estimator_.feature_importances_):
				if importance > 0:
					important_features.append(name)

			print(important_features)
			data_to_use[0] = data_to_use[0].loc[:, important_features]
		
		# TODO: take only selected number of best features

		# set optimized params for classifier
		self.model.clf.set_params(**self.model.optimization.best_params_)
		self.model.clf.fit(*data_to_use)

		return self
		
	def train_model(self):
		# TODO: implement
		pass


class PredictValuesBuilder(TrainModelBuilder):
	def predict(self, input_data = None):
		data_to_use = input_data if input_data else self.model.test_X
		
		self.model.pred_y = self.model.clf.predict(data_to_use)
		return self


"""class ValidateResultsBuilder(PredictValuesBuilder):
	def calc_metrics(\
		self,\
		average = ['binary', 'micro', 'macro', 'samples', 'weighted']\
	):
		for metric_name in ['accuracy', 'recall', 'f1', 'precision']:
			self.__calc_metric(metric_name)
            
		return self
    
	def __calc_metric(self, metric_name):
		gl = globals()
		metric_name = metric_name + '_score'
		kwargs = dict()
		if metric_name != 'accuracy_score':
			kwargs['average'] = 'macro'
            
        
		if 'sklearn' in gl:
			metric_fn = getattr(sklearn.metrics, metric_name)
			setattr(self.model, metric_name, metric_fn(self.model.test_y, self.model.pred_y, **kwargs))                
		elif metric_name in gl:
			setattr(self.model, metric_name, gl[metric_name](self.model.test_y, self.model.pred_y, **kwargs))                
		else:
			print(f'Metric {metric_name} is not imported.')"""
            

class ModelBuilder(PredictValuesBuilder):
	pass