import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.svm import NuSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from scipy import linalg, mat, dot;
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, roc_auc_score

seed = 8
np.random.seed(seed)

### data = pd.read_csv(".csv", header=None)

data = pd.read_csv("DWT_wavelets.csv",header=None)
data.head()
data.drop(data.columns[[0]], axis=1, inplace=True)
data.drop(data.head(1).index, inplace=True)
# create data frame into numpy array
###dataset = data.as_matrix()
dataset = data.to_numpy()
print(dataset.shape)
X = dataset[:, :-1]
Y = dataset[:, -1]

X = np.array(X)
print(X.shape)

Y = np.array(Y)

print(X.shape)
print(Y.shape)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# svm_class = SVC(kernel='linear')
# log_class = LogisticRegression()
# svm_class = SVC(kernel='poly', degree=8)
# svm_class = SVC(kernel='rbf')
# svm_class = SVC(kernel='sigmoid')
# svm_class = svm.LinearSVC()
# rf_class = RandomForestClassifier(n_estimators=10)
# rf_class = RandomForestClassifier(n_estimators=1000, random_state=0, max_features = 'auto')
# svm_class = NuSVC(kernel='rbf')

for M in range(2, 15, 2): # combines M trees
    for lr in [0.0001, 0.001, 0.01, 0.1, 1]:
        RFC_class = AdaBoostClassifier(n_estimators=M, learning_rate=lr, random_state=0)

a = cross_val_score(RFC_class, X, Y, scoring='accuracy', cv=kf)
y_pred = cross_val_predict(RFC_class, X, Y, cv=kf)
conf_mat = confusion_matrix(Y, y_pred)

sum = 0

for k in a:
    sum = sum + k

avg = 100 * (sum / 10)

print(a)
print(avg)

print(conf_mat)

TP = conf_mat[1, 1]
TN = conf_mat[0, 0]
FP = conf_mat[0, 1]
FN = conf_mat[1, 0]

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
classification_error = (FP + FN) / float(TP + TN + FP + FN)
sensitivity = TP / float(FN + TP)
specificity = TN / (TN + FP)
false_positive_rate = FP / float(TN + FP)
precision = TP / float(TP + FP)

print(classification_accuracy)
print(classification_error)
print(sensitivity)
print(specificity)
print(false_positive_rate)
print(precision)

report = classification_report(Y, y_pred)
print(report)

avg_auc = 0.0
tprs = []
base_fpr = np.linspace(0, 1, 101)

# to plot average roc curves of 10-fold
plt.figure(figsize=(5, 5))

for i, (train, test) in enumerate(kf.split(X, Y)):
    model = RFC_class.fit(X[train], Y[train])
    Y_score = model.predict_proba(X[test])
    fpr, tpr, _ = roc_curve(Y[test], Y_score[:, 1])
    auc = roc_auc_score(Y[test], Y_score[:, 1])
    avg_auc = avg_auc + auc

    plt.plot(fpr, tpr, 'b', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)

tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.plot(base_fpr, mean_tprs, 'b')
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.axes().set_aspect('equal', 'datalim')
plt.show()

avg_auc = avg_auc / 10
print("Area under the curve")
print('AUC: %.3f' % avg_auc)
