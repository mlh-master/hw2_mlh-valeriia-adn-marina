import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def find_distribution(X,features):
    count = 0
    feat = features.copy()
    feat.pop(0) #drop age
    feat.pop(15) #drop diagnosis
    percent = np.zeros(len(feat))
    for i in range(len(feat)):
        for j in X['Diagnosis'].index.values:
            if X[feat[i]][j] == 1:
                count += 1
                percent[i] = count/len(X[feat[i]])
        count = 0
    return percent*100,feat


def cv_kfold_logreg(X,y,C,K):
    kf = SKFold(n_splits=K)
    params = {'classifier': [LogisticRegression()],
              'classifier__penalty': ['l1', 'l2'],
              'classifier__C': C,
              'classifier__solver': ['liblinear']}
    pipe = Pipeline([('classifier', LogisticRegression())])
    logreg = GridSearchCV(estimator=pipe,
                          param_grid=params,
                          scoring=['roc_auc'],
                          cv=kf, refit='roc_auc', verbose=3, return_train_score=True)
    logreg.fit(X, y)
    best_logreg = logreg.best_estimator_
    return best_logreg


def cv_kfold_svm(X, y, C, K, gamma = [0], flag='linear'):
    kf = SKFold(n_splits=K)
    svc = svm.SVC(probability=True)
    pipe = Pipeline(steps=[('svm', svc)])
    if gamma == [0]:
        Svm = GridSearchCV(estimator=pipe,
                           param_grid={'svm__kernel': [flag], 'svm__C': C},
                           scoring=['roc_auc'],
                           cv=kf, refit='roc_auc', verbose=3, return_train_score=True)
    else:
        Svm = GridSearchCV(estimator=pipe,
                           param_grid={'svm__kernel': [flag], 'svm__C': C, 'svm__gamma': gamma},
                           scoring=['roc_auc'],
                           cv=kf, refit='roc_auc', verbose=3, return_train_score=True)

    Svm.fit(X, y)
    best_Svm = Svm.best_estimator_
    return best_Svm

def calc_stat(X, y,clf):
    # Taken from the tutorial
    calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
    calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
    calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
    calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)
    TN = calc_TN(y, y_pred)
    FP = calc_FP(y, y_pred)
    FN = calc_FN(y, y_pred)
    TP = calc_TP(y, y_pred)
    Se = TP / (TP + FN)
    PPV = TP / (TP + FP)
    Acc = (TP + TN) / (TP + TN + FP + FN)
    F1 = (2 * Se * PPV) / (Se + PPV)
    AUROC = roc_auc_score(y, y_pred_proba[:, 1])
    return Acc, F1, AUROC