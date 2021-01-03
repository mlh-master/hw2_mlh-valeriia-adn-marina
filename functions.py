import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.metrics import log_loss
from sklearn import svm
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hinge_loss

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



def loss_svm(W, X, y, regularization=0.001):
    loss = 0.0
    delta = 1.0
    train = y.shape[0]
    scores_matrix = W.dot(X)
    correct = scores_matrix[y, xrange(train)]
    margins = scores_matrix - correct + delta
    margins = np.maximum(0, margins)
    margins[y, xrange(train)] = 0
    loss = np.sum(margins) / train
    loss += 0.5 * regularization * np.sum(W * W)
    return loss

def cv_kfold_logreg(X,y,C,K):
    kf = SKFold(n_splits=K)
    svc = svm.SVC(probability=True)
    params = {'classifier': [LogisticRegression()],
              'classifier__penalty': ['l1', 'l2'],
              'classifier__C': C,
              'classifier__solver': ['liblinear']}
    pipe = Pipeline([('classifier', LogisticRegression())])
    logreg = GridSearchCV(estimator=pipe,
                          param_grid=params,
                          scoring=['roc_auc'],
                          cv=kf, refit='roc_auc', verbose=3, return_train_score=True)
    for train_idx, val_idx in kf.split(X, y):
        x_train, x_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        logreg.fit(x_train, y_train)
        y_pred = logreg.predict_proba(x_val)
    return y_pred, logreg


def cv_kfold_svm(X, y, C, K,gamma, flag ='linear'):
    kf = SKFold(n_splits=K)
    svc = svm.SVC(probability=True)
    pipe = Pipeline(steps=[('svm', svc)])
    svm_lin = GridSearchCV(estimator=pipe,
                           param_grid={'svm__kernel': [flag], 'svm__C': C, 'svm__gamma': gamma},
                           scoring=['roc_auc'],
                           cv= kf, refit='roc_auc', verbose=3, return_train_score=True)
    for train_idx, val_idx in kf.split(X, y):
        x_train, x_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        svm_lin.fit(x_train, y_train)
        y_pred = svm_lin.predict_proba(x_val)
    return y_pred,svm_lin

def calc_stat(X_test, y_test,logreg, flag = 'logreg', W = 0):

    calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
    calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
    calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
    calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]
    y_pred_test = logreg.predict(X_test)
    y_pred_proba_test = logreg.predict_proba(X_test)
    TN = calc_TN(y_test, y_pred_test)
    FP = calc_FP(y_test, y_pred_test)
    FN = calc_FN(y_test, y_pred_test)
    TP = calc_TP(y_test, y_pred_test)
    Se = TP / (TP + FN)
    Sp = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    Acc = (TP + TN) / (TP + TN + FP + FN)
    F1 = (2 * Se * PPV) / (Se + PPV)

    if flag == 'logreg':
        loss = log_loss(y_test, y_pred_proba_test)
    else:
        # loss = hinge_loss(y_test,y_pred_proba_test)
        loss = loss_svm(W,X_test,y_test)
    print('Loss is {:.2f}. \nAccuracy is {:.2f}. \nF1 is {:.2f}. '.format(loss, Acc, F1))
    print('AUROC is {:.3f}'.format(roc_auc_score(y_test, y_pred_proba_test[:, 1])))
