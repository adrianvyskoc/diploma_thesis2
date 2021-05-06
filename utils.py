from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score   
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
import numpy as np

def make_stratified_k_fold(n_splits, clf, X, y):
	
	best = 0
	RFC_dict = {
		"accuracies": [],
		"precisions": [],
		"recalls":[],
		"f1_scores": []
	}
	
	skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=True)
	
	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
	
		print("{:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\"
			.format(
				precision_score(y_test, y_pred, average='macro'), 
				recall_score(y_test, y_pred, average='macro'),
				accuracy_score(y_test, y_pred),
				f1_score(y_test, y_pred, average='macro')
			)
		)
	
		RFC_dict['accuracies'].append(accuracy_score(y_test, y_pred))
		RFC_dict['f1_scores'].append(f1_score(y_test, y_pred, average='macro'))
		RFC_dict['recalls'].append(recall_score(y_test, y_pred, average='macro'))
		RFC_dict['precisions'].append(precision_score(y_test, y_pred, average='macro'))
	
	def Average(lst): 
		return sum(lst) / len(lst) 
	
	print("{:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\"
		.format(
			Average(RFC_dict['precisions']),
			Average(RFC_dict['recalls']),
			Average(RFC_dict['accuracies']),
			Average(RFC_dict['f1_scores'])
		)
	)

    
def create_roc_curve(n_splits, clf, X, y):
	# Run classifier with cross-validation and plot ROC curves
	cv = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=True)
	
	tprs = []
	aucs = []
	mean_fpr = np.linspace(0, 1, 100)
	
	fig, ax = plt.subplots()
	for i, (train, test) in enumerate(cv.split(X, y)):
		X_train, X_test = X.iloc[train], X.iloc[test]
		y_train, y_test = y.iloc[train], y.iloc[test]
		
		clf.fit(X_train, y_train)
		viz = plot_roc_curve(clf, X_test, y_test, name='ROC fold {}'.format(i), alpha=0.3, lw=1, ax=ax)
		interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
		interp_tpr[0] = 0.0
		tprs.append(interp_tpr)
		aucs.append(viz.roc_auc)
	
	ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
	
	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)
	ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
	
	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
	
	ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic example")
	ax.legend(loc="lower right")
	plt.show()