import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from mlxtend.classifier import EnsembleVoteClassifier
import matplotlib.pyplot as plt
#from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb
import catboost as cbt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

data = pd.read_csv('train.csv')
#########################feature engineer##########################
data['Attrition'] = data['Attrition'].map({'Yes':1,'No':0})
data['YearsAtCompany_Less_One_or_More_20'] = [1 if (value <= 1 or value >=20) else 0 for value in data['YearsAtCompany']]
data['YearsInCurrentRole_Less_2'] = [1 if value <= 2 else 0 for value in data['YearsInCurrentRole']]
data['YearsSinceLastPromotion_Less_1_or_More_6'] = [1 if (value < 1 or value >=6) else 0 for value in data['YearsSinceLastPromotion']]
data['YearsWithCurrManager_Less_1_or_More_11'] = [1 if (value < 1 or value >=11) else 0 for value in data['YearsWithCurrManager']]
data['Age_Less_22'] = [1 if value < 22 else 0 for value in data['Age']]
data['BusinessTravel_Frequently'] = [1 if str(value) == 'Travel_Frequently' else 0 for value in data['BusinessTravel']]
data['DistanceFromHome_Less_12_More_27'] = [1 if (value >= 12 or value <=27) else 0 for value in data['DistanceFromHome']]
data['WorkLifeBalance_1'] = [1 if value == 1 else 0 for value in data['WorkLifeBalance']]
data['TotalWorkingYears_Less_2'] = [1 if value <= 2 else 0 for value in data['TotalWorkingYears']]
data['EducationField_HR'] = [1 if str(value) =='Human Resources' else 0 for value in data['EducationField']]
data['MaritalStatus_Single'] = [1 if str(value) == 'Single' else 0 for value in data['MaritalStatus']]
data['EnvironmentSatisfaction_Less_1'] = [1 if value < 1.5 else 0 for value in data['EnvironmentSatisfaction']]
data['JobInvolvement_Less_1'] = [1 if value < 1.5 else 0 for value in data['JobInvolvement']]
data['NumCompaniesWorked_More_4'] = [1 if value > 4.5 else 0 for value in data['NumCompaniesWorked']]
data['JobSatisfaction_Less_1'] = [1 if value < 1.5 else 0 for value in data['JobSatisfaction']]
data['OverTime_Yes'] = [1 if str(value) == 'Yes' else 0 for value in data['OverTime']]
data['StockOptionLevel_Less_0'] = [1 if value < 1 else 0 for value in data['StockOptionLevel']]
data['TrainingTimesLastYear_Less_0'] = [1 if value < 1 else 0 for value in data['TrainingTimesLastYear']]
data['TotalWorkingYears_Less_1_More_40'] = [1 if value < 2 or value > 39 else 0 for value in data['TotalWorkingYears']]
print("feature engineer data shape:", data.shape)
###########################feature engineer##########################
y_labels = data['Attrition']
features = data.drop(['user_id', 'Attrition'], axis=1)
columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
le = LabelEncoder()
for column in columns:
    le.fit(features[column])
    features[column] = le.transform(features[column])

ss = MinMaxScaler()
features_ss = ss.fit_transform(features)
x_train, x_test, y_train, y_test = train_test_split(features_ss, y_labels, test_size=0.25, random_state=1457)

#print(data.groupby(['Gender','Age']).agg({'Attrition':'mean'}))
# data_1 = data[data.Attrition == 'Yes']
# data1_virtual = data_1.sample(n= 300,replace=True)
# data1_virtual.index = range(1176, 1476)
# data = pd.concat([data,data1_virtual],axis=0)
#data['YearsAtCompany_Less_One_or_More_20'] = [1 if value <= 1 else 0 for value in data['YearsAtCompany']]
#data['YearsAtCompany_Less_Four'] = [1 if value <= 4 else 0 for value in data['YearsAtCompany']]
#excute_columns = ['user_id', 'Attrition']
#features = data[[column for column in data.columns if column not in excute_columns]]
# sm = SMOTE(random_state=42, n_jobs=-1)
# x, y = sm.fit_resample(features, y_labels)
# np.savetxt('x.txt', x)
# np.savetxt('y.txt', y)
#x = pd.read_table('x.txt', header=None, delim_whitespace=True)
#y = pd.read_table('y.txt', header=None, delim_whitespace=True)


def plot_roc_curve(y_pred, y_pred_score):
    fpr, tpr, thresholds = roc_curve(y_pred, y_pred_score)
    roc_auc = auc(fpr, tpr)
    # Plot ROC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.0])
    plt.ylim([-0.1, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()
    print("roc_auc:", roc_auc)


def cross_validation(clf, x1, y1, k):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, x1, y1, cv=k, scoring='roc_auc')
    print(scores)
    print(np.mean(scores))


def call_lgb():
    print("####lightgbm*****")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=55, reg_alpha=0.0, reg_lambda=1,
        max_depth=15, n_estimators=6000, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=4
    )
    cross_validation(clf, 5)
    clf.fit(x_train, y_train)
    y_pre = clf.predict_proba(x_test)
    y_pre = y_pre[:, 1].reshape(-1, 1)
    plot_roc_curve(y_test, y_pre)
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
    cross_validation(xg, 5)
    xg.fit(x_train, y_train)
    y_pre = xg.predict_proba(x_test)
    y_pre = y_pre[:, 1].reshape(-1, 1)
    plot_roc_curve(y_test, y_pre)


def call_LR():
    print("####LogisticRegression*****")
    lr = LogisticRegression()
    cross_validation(lr, 5)
    lr.fit(x_train, y_train)
    y_pre = lr.predict_proba(x_test)
    y_pre = y_pre[:, 1].reshape(-1, 1)
    plot_roc_curve(y_test, y_pre)
    # print(metrics.roc_auc_score(y_test, y_pre))
    # print(np.mean(np.equal(y_pre, y_train)))
    test_data = pd.read_csv('test.csv')
    test_data['YearsAtCompany_Less_One'] = [1 if value <= 1 else 0 for value in test_data['YearsAtCompany']]
    test_data['YearsAtCompany_Less_Four'] = [1 if value <= 4 else 0 for value in test_data['YearsAtCompany']]
    print("test_data shape:{}".format(test_data.shape))
    pre_data = test_data.drop(['user_id'], axis=1)
    for column in columns:
        pre_data[column] = le.fit_transform(pre_data[column])
    pre_data = ss.transform(pre_data)
    y_pre = lr.predict_proba(pre_data)
    y_res = []
    for item in y_pre[:, 1]:
        y_res.append(item)
    pre = pd.DataFrame({'user_id': test_data['user_id'], 'Attrition': y_res}, index=None)
    pre.to_csv('pre_lr_features.csv', index=None)


def call_rf():
    print("####RandomForest*****")
    rf = RandomForestClassifier(n_jobs=4)
    cross_validation(rf, 5)
    rf.fit(x_train, y_train)
    y_pre = rf.predict_proba(x_test)
    y_pre = y_pre[:, 1].reshape(-1, 1)
    plot_roc_curve(y_test, y_pre)


def call_catboost():
    print("####catboost*****")
    glf = cbt.CatBoostClassifier(
        iterations=2,
        learning_rate=1,
        depth=5, loss_function='Logloss')
    cross_validation(glf, 5)
    glf.fit(x_train, y_train)
    y_pre = glf.predict_proba(x_test)
    y_pre = y_pre[:, 1].reshape(-1, 1)
    plot_roc_curve(y_test, y_pre)


def model_ensemble_stacking(x_train, y_train, x_test, y_test):
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
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=55, reg_alpha=0.0, reg_lambda=1,
        max_depth=15, n_estimators=6000, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.2, min_child_weight=1, random_state=20, n_jobs=4
    )
    glf = cbt.CatBoostClassifier(depth=2, loss_function='Logloss')
    rf = RandomForestClassifier(n_jobs=4)
    clfs = [xg, glf,clf]
    print("*******start train*******")
    print("x_train shape :{} y_train shape:{}".format(x_train.shape, y_train.shape))
    x_train = pd.DataFrame(data=x_train, columns=features.columns)
    for index, clf in enumerate(clfs):
        clf.fit(x_train, y_train)
    print("*******end train*******")
    x_all_train = x_train
    for index, clf in enumerate(clfs):
        y_res = clf.predict_proba(x_train)
        y_res = [item[1] for index, item in enumerate(y_res)]
        y_res = np.array(y_res)
        y_res = y_res.reshape(-1, 1)
        x_all_train = np.hstack((x_all_train, y_res))
    print(x_all_train.shape, y_train.shape)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(x_all_train, y_train)
    y_pre = lr.predict_proba(x_all_train)
    y_pre = y_pre[:, 1].reshape(-1, 1)
    plot_roc_curve(y_train, y_pre)
    print("********Test on test data*************")
    x_test = pd.DataFrame(data=x_test, columns=features.columns)
    x_all_test = x_test
    print("x_test shape :{}".format(x_test.shape))
    for index, clf in enumerate(clfs):
        y_res = clf.predict_proba(x_test)
        y_res = [item[1] for index, item in enumerate(y_res)]
        y_res = np.array(y_res)
        y_res = y_res.reshape(-1, 1)
        x_all_test = np.hstack((x_all_test, y_res))
    print("x_all_test shape:{}".format(x_all_test.shape))
    y_pre = lr.predict_proba(x_all_test)
    y_pre = y_pre[:, 1].reshape(-1, 1)
    plot_roc_curve(y_test, y_pre)
    cross_validation(lr, x_all_test, y_test, 5)
    # #
    # print("Read really test data")
    #     # test_data = pd.read_csv('test.csv')
    #     # test_data['YearsAtCompany_Less_One'] = [1 if value <= 1 else 0 for value in test_data['YearsAtCompany']]
    #     # test_data['YearsAtCompany_Less_Four'] = [1 if value <= 4 else 0 for value in test_data['YearsAtCompany']]
    #     # pre_data = test_data.drop(['user_id'], axis=1)
    #     # for column in columns:
    #     #     pre_data[column] = le.fit_transform(pre_data[column])
    #     # print(pre_data.columns)
    #     # pre_data = ss.transform(pre_data)
    #     # pre_data = pd.DataFrame(data=pre_data, columns=features.columns)
    #     # print(pre_data.columns)
    #     # test_data_all = pre_data
    #     # print("pre_data shape:", pre_data.shape)
    #     # for index, clf in enumerate(clfs):
    #     #     print("index = {}".format(index))
    #     #     y_res = clf.predict_proba(pre_data)
    #     #     y_res = [item[1] for index, item in enumerate(y_res)]
    #     #     y_res = np.array(y_res)
    #     #     y_res = y_res.reshape(-1, 1)
    #     #     test_data_all = np.hstack((test_data_all, y_res))
    #     # print("test_data_all shape:{}".format(test_data_all.shape))
    #     # y_res = lr.predict_proba(test_data_all)
    #     # pre = pd.DataFrame({'user_id': test_data['user_id'], 'Attrition': y_res[:, 1]}, index=None)
    #     # pre.to_csv('pre_ensemble_features.csv', index=None)

def model_ensemble_voting(x_train,y_train,x_test,y_test):
    xg = xgb.XGBClassifier(
        booster='gbtree',objective='binary:logistic',seed=2020
    )

    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=55, reg_alpha=0.0, reg_lambda=1,
        max_depth=15, n_estimators=6000, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.2, min_child_weight=1, random_state=20, n_jobs=4
    )
    glf = cbt.CatBoostClassifier(depth=2, loss_function='Logloss')
    rf = RandomForestClassifier(n_jobs=4)
    lr = LogisticRegression(max_iter=1000)
    clfs = [xg, clf, glf, rf, lr]
    print("*******start train*******")
    # print("x_train shape :{} y_train shape:{}".format(x_train.shape, y_train.shape))
    # x_train = pd.DataFrame(data=x_train,columns=features.columns)
    # for clf in clfs:
    #     clf.fit(x_train, y_train)
    # print("********Test on test data*************")
    # x_test = pd.DataFrame(data=x_test, columns=features.columns)
    # y_vote_value = []
    # for clf in clfs:
    #     y_pre = clf.predict_proba(x_test)
    #     y_pre = y_pre[:, 1].reshape(-1, 1)
    #     plot_roc_curve(y_test, y_pre)
    #     y_vote_value.append(y_pre)
    x_train = pd.DataFrame(data=x_train, columns=features.columns)
    vote_class = EnsembleVoteClassifier(clfs=clfs,weights=[2,1,1,1,5],voting='soft')
    vote_class.fit(x_train,y_train)
    x_test = pd.DataFrame(data=x_test,columns=features.columns)
    y_pre = vote_class.predict_proba(x_test)
    y_pre = y_pre[:, 1].reshape(-1, 1)
    plot_roc_curve(y_test,y_pre)
    print("********Predict on really data*************")
    data = pd.read_csv('test.csv')
    data['YearsAtCompany_Less_One_or_More_20'] = [1 if (value <= 1 or value >= 20) else 0 for value in
                                                  data['YearsAtCompany']]
    data['YearsInCurrentRole_Less_2'] = [1 if value <= 2 else 0 for value in data['YearsInCurrentRole']]
    data['YearsSinceLastPromotion_Less_1_or_More_6'] = [1 if (value < 1 or value >=6) else 0 for value in data['YearsSinceLastPromotion']]
    data['YearsWithCurrManager_Less_1_or_More_11'] = [1 if (value < 1 or value >=11) else 0 for value in data['YearsWithCurrManager']]
    data['Age_Less_22'] = [1 if value < 22 else 0 for value in data['Age']]
    data['BusinessTravel_Frequently'] = [1 if str(value) == 'Travel_Frequently' else 0 for value in data['BusinessTravel']]
    data['DistanceFromHome_Less_12_More_27'] = [1 if (value >= 12 or value <=27) else 0 for value in data['DistanceFromHome']]
    data['WorkLifeBalance_1'] = [1 if value == 1 else 0 for value in data['WorkLifeBalance']]
    data['TotalWorkingYears_Less_2'] = [1 if value <= 2 else 0 for value in data['TotalWorkingYears']]
    data['EducationField_HR'] = [1 if str(value) =='Human Resources' else 0 for value in data['EducationField']]
    data['MaritalStatus_Single'] = [1 if str(value) == 'Single' else 0 for value in data['MaritalStatus']]
    data['EnvironmentSatisfaction_Less_1'] = [1 if value < 1.5 else 0 for value in data['EnvironmentSatisfaction']]
    data['JobInvolvement_Less_1'] = [1 if value < 1.5 else 0 for value in data['JobInvolvement']]
    data['NumCompaniesWorked_More_4'] = [1 if value > 4.5 else 0 for value in data['NumCompaniesWorked']]
    data['JobSatisfaction_Less_1'] = [1 if value < 1.5 else 0 for value in data['JobSatisfaction']]
    data['OverTime_Yes'] = [1 if str(value) == 'Yes' else 0 for value in data['OverTime']]
    data['StockOptionLevel_Less_0'] = [1 if value < 1 else 0 for value in data['StockOptionLevel']]
    data['TrainingTimesLastYear_Less_0'] = [1 if value < 1 else 0 for value in data['TrainingTimesLastYear']]
    data['TotalWorkingYears_Less_1_More_40'] = [1 if value < 2 or value > 39 else 0 for value in
                                                data['TotalWorkingYears']]
    print("test_data shape:{}".format(data.shape))
    pre_data = data.drop(['user_id'], axis=1)
    for column in columns:
        pre_data[column] = le.fit_transform(pre_data[column])
    pre_data = ss.transform(pre_data)
    pre_data = pd.DataFrame(data=pre_data, columns=features.columns)
    y_pre = vote_class.predict_proba(pre_data)
    y_res = []
    for item in y_pre[:, 1]:
        y_res.append(item)
    pre = pd.DataFrame({'user_id': data['user_id'], 'Attrition': y_res}, index=None)
    pre.to_csv('pre_vote_more_features.csv', index=None)

#########  focal loss ###########

if __name__ == '__main__':
    print("start")
    # call_LR()
    # call_xgboost()
    # call_lgb()
    # call_rf()
    # call_catboost()
    #model_ensemble_stacking(x_train, y_train, x_test, y_test)
    model_ensemble_voting(x_train,y_train,x_test,y_test)
