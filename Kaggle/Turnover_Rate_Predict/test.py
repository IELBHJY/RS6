import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

import xgboost as xgb
#import lightgbm as lgb
from lightgbm import LGBMClassifier
import catboost as cbt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

data = pd.read_csv('train.csv')

#data_1 = data[data.Attrition == 'Yes']
#data1_virtual = data_1.sample(n= 300,replace=True)
#data1_virtual.index = range(1176, 1476)
#data = pd.concat([data,data1_virtual],axis=0)
y_labels = data['Attrition']
features = data.drop(['user_id','Attrition'],axis = 1)
columns=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']
le = LabelEncoder()
for column in columns:
    le.fit(features[column])
    features[column] = le.transform(features[column])
le.fit(y_labels)
y_labels = le.transform(y_labels)
ss = MinMaxScaler()
features = ss.fit_transform(features)

sm = SMOTE(random_state= 42, n_jobs=-1)
x,y = sm.fit_resample(features,y_labels)
#print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=1457)

def plot_roc_curve(y_pred, y_pred_score):
    fpr, tpr, thresholds = roc_curve(y_pred, y_pred_score)
    roc_auc = auc(fpr, tpr)
    # Plot ROC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.0])
    plt.ylim([-0.1,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    print(roc_auc)

def cross_validation(clf,k):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf,x_train,y_train,cv=k,scoring='roc_auc')
    print(scores)
    print(np.mean(scores))

def call_lgb():
    print("####lightgbm*****")
    clf = LGBMClassifier(objective='bin')
    cross_validation(clf, 5)
    # clf.fit(x_train,y_train)
    # y_pre = clf.predict_proba(x_test)
    # y_pre = y_pre[:, 1].reshape(-1, 1)
    # plot_roc_curve(y_test,y_pre)
    # print(y_pre.shape, y_test.shape)
    # print(metrics.roc_auc_score(y_test, y_pre))
    # test_data = pd.read_csv('test.csv')
    # pre_data = test_data.drop(['user_id'], axis=1)
    # for column in columns:
    #     pre_data[column] = le.fit_transform(pre_data[column])
    # pre_data = ss.transform(pre_data)
    # y_pre = clf.predict_proba(pre_data)
    # y_res = []
    # for item in y_pre[:, 1]:
    #     y_res.append(item)
    # pre = pd.DataFrame({'user_id': test_data['user_id'], 'Attrition': y_res}, index=None)
    # pre.to_csv('pre_lgb_sample.csv',index=None)

def call_xgboost():
    print("####xgboost*****")
    xgboost_params ={
        'booster':'gbtree',
        'min_child_weight': 100,
        'eta': 0.02,
        'colsample_bytree': 0.7,
        'max_depth': 12,
        'subsample': 0.7,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'objective': 'reg:linear',
        'verbose_eval': True,
        'eval_metric':'auc',
        'seed': 12}
    xg = xgb.XGBClassifier(**xgboost_params)
    cross_validation(xg,5)
    #xg.fit(x_train, y_train)
    # y_pre = xg.predict_proba(x_test)
    # y_pre = y_pre[:, 1].reshape(-1, 1)
    # plot_roc_curve(y_test,y_pre)

def call_LR():
    print("####LogisticRegression*****")
    lr = LogisticRegression()
    cross_validation(lr,5)
    #lr.fit(x_train, y_train)
    # y_pre = lr.predict_proba(x_test)
    # y_pre = y_pre[:,1].reshape(-1,1)
    #plot_roc_curve(y_test,y_pre)
    #print(metrics.roc_auc_score(y_test, y_pre))
    #print(np.mean(np.equal(y_pre, y_train)))
    # test_data = pd.read_csv('test.csv')
    # print("test_data shape:{}".format(test_data.shape))
    # pre_data = test_data.drop(['user_id'], axis=1)
    # for column in columns:
    #     pre_data[column] = le.fit_transform(pre_data[column])
    # pre_data = ss.transform(pre_data)
    # y_pre = lr.predict_proba(pre_data)
    # y_res =[]
    # for item in y_pre[:,1]:
    #     y_res.append(item)
    # pre = pd.DataFrame({'user_id': test_data['user_id'], 'Attrition': y_res}, index=None)
    # pre.to_csv('pre_lr_prob_sample_data.csv', index=None)

def call_rf():
    print("####RandomForest*****")
    rf = RandomForestClassifier(n_jobs=4)
    cross_validation(rf,5)
    #rf.fit(x_train, y_train)
    # y_pre = rf.predict_proba(x_test)
    # y_pre = y_pre[:, 1].reshape(-1, 1)
    # plot_roc_curve(y_test,y_pre)

def call_catboost():
    print("####catboost*****")
    glf = cbt.CatBoostClassifier(
        iterations=2,
        learning_rate=1,
        depth=5, loss_function='Logloss')
    cross_validation(glf,5)
    #glf.fit(x_train,y_train)
    # y_pre = glf.predict_proba(x_test)
    # y_pre = y_pre[:, 1].reshape(-1, 1)
    # plot_roc_curve(y_test, y_pre)

def model_ensemble(x_train, y_train, x_test, y_test):
    xgboost_params = {
        'booster': 'gbtree',
        'min_child_weight': 100,
        'eta': 0.02,
        'colsample_bytree': 0.7,
        'max_depth': 12,
        'subsample': 0.7,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'objective': 'reg:linear',
        'verbose_eval': True,
        'eval_metric': 'auc',
        'seed': 12}
    xg = xgb.XGBClassifier(**xgboost_params)

    # clf = lgb.LGBMClassifier(
    #     boosting_type='gbdt', num_leaves=55, reg_alpha=0.0, reg_lambda=1,
    #     max_depth=15, n_estimators=6000, objective='binary',
    #     subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
    #     learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4
    # )
    glf = cbt.CatBoostClassifier(
        iterations=2,
        learning_rate=1,
        depth=2, loss_function='Logloss')
    rf = RandomForestClassifier(n_jobs=4)
    clfs=[xg, glf, rf]
    for index, clf in enumerate(clfs):
        clf.fit(x_train, y_train)

    x_all = x_test
    for index, clf in enumerate(clfs):
        y_res = clf.predict_proba(x_test)
        y_res = [item[1] for index, item in enumerate(y_res)]
        y_res = np.array(y_res)
        y_res = y_res.reshape(-1, 1)
        x_all = np.hstack((x_all, y_res))
    lr = LogisticRegression()
    lr.fit(x_all, y_test)
    y_pre = lr.predict_proba(x_all)
    y_pre = y_pre[:, 1].reshape(-1, 1)
    plot_roc_curve(y_test, y_pre)

    test_data = pd.read_csv('test.csv')
    pre_data = test_data.drop(['user_id'], axis=1)
    for column in columns:
        pre_data[column] = le.fit_transform(pre_data[column])
    pre_data = ss.transform(pre_data)
    print(test_data.shape)
    test_data_all = pre_data
    for index, clf in enumerate(clfs):
        y_res = clf.predict_proba(pre_data)
        y_res = [item[1] for index, item in enumerate(y_res)]
        y_res = np.array(y_res)
        y_res = y_res.reshape(-1, 1)
        test_data_all = np.hstack((test_data_all, y_res))
    #print(test_data_all)
    y_res = lr.predict_proba(test_data_all)
    pre = pd.DataFrame({'user_id': test_data['user_id'], 'Attrition': y_res[:, 1]}, index=None)
    pre.to_csv('pre_ensemble_smote.csv', index=None)


#########  focal loss ###########

if __name__ == '__main__':
    call_LR()
    #call_xgboost()
    #call_lgb()
    #call_rf()
    #call_catboost()
    #model_ensemble(x_train,y_train,x_test,y_test)