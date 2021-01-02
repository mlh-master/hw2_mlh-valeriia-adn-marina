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


def cv_kfold_logreg(X, y, C, penalty, K):
    """

    :param X: Training set samples
    :param y: Training set labels
    :param C: A list of regularization parameters
    :param penalty: A list of types of norm
    :param K: Number of folds
    :param mode: Mode of normalization (parameter of norm_standard function in clean_data module)
    :return: A dictionary as explained in the notebook
    """
    kf = SKFold(n_splits=K)
    validation_dict = []

    f = {}
    for c in C:
        for p in penalty:
            clf = LogisticRegression(solver='saga', penalty=p, C=c, max_iter=10000, multi_class='ovr')
            loss_val_vec = np.zeros(K)
            k = 0
            for train_idx, val_idx in kf.split(X, y):
                x_train, x_val = X[train_idx], X[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                clf.fit(x_train, y_train)
                v = clf.predict_proba(x_val)
                loss_val_vec[k] = log_loss(y_val, v)
                k += 1
            mu = loss_val_vec.mean()

            std = loss_val_vec.std()

            validation_dict.append({'mu': mu, 'sigma': std, 'C': c, 'penalty': p})

    return validation_dict



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

def calc_stat(X_test, y_test,logreg):

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
    loss = log_loss(y_test, y_pred_proba_test)
    print('Loss is {:.2f}. \nAccuracy is {:.2f}. \nF1 is {:.2f}. '.format(loss, Acc, F1))
    print('AUROC is {:.3f}'.format(roc_auc_score(y_test, y_pred_proba_test[:, 1])))

def prob_curves(val_dict,flag):

    for d in val_dict:
        x = np.linspace(0, d['mu'] + 3 * d['sigma'], 1000)
        if flag == 'logreg':
            label = "p = " + d['penalty'] + ", C = " + str(d['C'])
        else:
            label = "gamma = " + str(d['gamma']) + ", C = " + str(d['C'])
        plt.plot(x, stats.norm.pdf(x, d['mu'], d['sigma']), label=label)
        plt.title('Gaussian distribution of the loss')
        plt.xlabel('Average loss')
        plt.ylabel('Probabilty density')
    plt.legend()
    plt.show()