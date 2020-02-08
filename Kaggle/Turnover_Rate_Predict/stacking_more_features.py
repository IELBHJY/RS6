import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.combine import SMOTETomek
import lightgbm as lgb
import catboost as cbt
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

pd.set_option('display.max_columns', None)

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
excute_columns = ['user_id', 'Attrition']
target = ['Attrition']
origin_columns = [column for column in train_data.columns if column not in excute_columns]


def feature_engineer(data):
    print("input data shape:", data.shape)
    data['YearsAtCompany_Less_One_or_More_20'] = [1 if (value <= 1 or value >= 20) else 0 for value in
                                                  data['YearsAtCompany']]
    data['YearsInCurrentRole_Less_2'] = [1 if value <= 2 else 0 for value in data['YearsInCurrentRole']]
    data['YearsSinceLastPromotion_Less_1_or_More_6'] = [1 if (value < 1 or value >= 6) else 0 for value in
                                                        data['YearsSinceLastPromotion']]
    data['YearsWithCurrManager_Less_1_or_More_11'] = [1 if (value < 1 or value >= 11) else 0 for value in
                                                      data['YearsWithCurrManager']]
    data['Age_Less_22'] = [1 if value < 22 else 0 for value in data['Age']]
    data['BusinessTravel_Frequently'] = [1 if str(value) == 'Travel_Frequently' else 0 for value in
                                         data['BusinessTravel']]
    data['DistanceFromHome_Less_12_More_27'] = [1 if (value >= 12 or value <= 27) else 0 for value in
                                                data['DistanceFromHome']]
    data['WorkLifeBalance_1'] = [1 if value == 1 else 0 for value in data['WorkLifeBalance']]
    data['TotalWorkingYears_Less_2'] = [1 if value <= 2 else 0 for value in data['TotalWorkingYears']]
    data['EducationField_HR'] = [1 if str(value) == 'Human Resources' else 0 for value in data['EducationField']]
    #second-stage
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
    return data


train = feature_engineer(train_data)
test = feature_engineer(test_data)

features_columns = [column for column in train.columns if column not in excute_columns]
columns = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18',
           'OverTime']


def preprocess(data):
    features = data[[column for column in data.columns if column not in excute_columns]]
    print(features.shape)
    le = LabelEncoder()
    for column in columns:
        le.fit(features[column])
        features[column] = le.transform(features[column])
    ss = MinMaxScaler()
    features_ss = ss.fit_transform(features)
    if 'Attrition' not in data.columns:
        return features_ss
    else:
        y_labels = data['Attrition'].map({'Yes': 1, 'No': 0})
        return features_ss, y_labels


x_train, y_train = preprocess(train)
x_test = preprocess(test)


smt = SMOTE(random_state=2020,k_neighbors=3)
x_train, y_train = smt.fit_sample(x_train, y_train)


x_train_multi = pd.DataFrame(data=x_train, columns=features_columns)
y_train_multi = pd.DataFrame(data=y_train, columns=target)
x_test_multi = pd.DataFrame(data=x_test, columns=features_columns)
ntrain = x_train_multi.shape[0]
ntest = x_test_multi.shape[0]
print(ntrain, ntest)

NFOLDS = 5  # the variable for cross-validation folds
SEED = 2020
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
#kf = StratifiedKFold(n_splits=NFOLDS, random_state=SEED)


def get_oof(clf, ntrain, ntest, x_train,
            y_train):  # out of fold, which means each step use the k-fold way to cross validate the dataset.
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        if isinstance(x_train, np.ndarray):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]
        else:
            x_tr = x_train.iloc[train_index]
            y_tr = y_train.iloc[train_index]
            x_te = x_train.iloc[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test_multi)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]


class CatboostWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_seed'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]


class LightGBMWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['feature_fraction_seed'] = seed
        params['bagging_seed'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


et_params = {
    'n_jobs': 16,
    'n_estimators': 600,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
    #     'silent': 1,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 600,
    'max_features': 0.2,
    'max_depth': 12,
    'min_samples_leaf': 2,
    #     'silent': 1,
}

xgb_params = {
    'booster': 'gbtree',
    'seed': 2020,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.75,
    'learning_rate': 0.075,
    'objective': 'binary:logistic',
    'max_depth': 7,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'nrounds': 200
}

catboost_params = {
    'iterations': 600,
    'learning_rate': 0.5,
    'depth': 10,
    'l2_leaf_reg': 40,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.7,
    'scale_pos_weight': 5,
    'eval_metric': 'AUC',
    'od_type': 'Iter',
    'allow_writing_files': False,
    'silent': True,
}

lightgbm_params = {
    'n_estimators': 600,
    'learning_rate': 0.1,
    'num_leaves': 123,
    'colsample_bytree': 0.8,
    'subsample': 0.9,
    'max_depth': 15,
    'reg_alpha': 0.2,
    'reg_lambda': 0.4,
    'min_split_gain': 0.01,
    'min_child_weight': 2,
    'silent': 1,
}
xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
cb = CatboostWrapper(clf=CatBoostClassifier, seed=SEED, params=catboost_params)
#lg = LightGBMWrapper(clf=LGBMClassifier, seed=SEED, params=lightgbm_params)
xg_oof_train, xg_oof_test = get_oof(xg, ntrain, ntest, x_train_multi, y_train_multi)
et_oof_train, et_oof_test = get_oof(et, ntrain, ntest, x_train_multi, y_train_multi)
rf_oof_train, rf_oof_test = get_oof(rf, ntrain, ntest, x_train_multi, y_train_multi)
cb_oof_train, cb_oof_test = get_oof(cb, ntrain, ntest, x_train_multi, y_train_multi)
#lg_oof_train, lg_oof_test = get_oof(lg, ntrain, ntest, x_train_multi, y_train_multi)
print("cv over")
print("XG-CV: {}".format(roc_auc_score(y_train_multi, xg_oof_train)))
print("ET-CV: {}".format(roc_auc_score(y_train_multi, et_oof_train)))
print("RF-CV: {}".format(roc_auc_score(y_train_multi, rf_oof_train)))
print("CB-CV: {}".format(roc_auc_score(y_train_multi, cb_oof_train)))
#print("LG-CV:{}".format(roc_auc_score(y_train_multi, lg_oof_train)))

x_train_fin = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, cb_oof_train), axis=1)
x_test_fin = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, cb_oof_test), axis=1)
print("{},{}".format(x_train_fin.shape, x_test_fin.shape))
lr = LogisticRegression()
lr.fit(x_train_fin, y_train_multi)
result = pd.DataFrame()
result['user_id'] = test['user_id']
result['Attrition'] = lr.predict_proba(x_test_fin)[:, 1]
result.to_csv('results/stacking_more_features_smote.csv', index=None)
# print(result)
# def cross_validation(clf, x1, y1, k):
#     from sklearn.model_selection import cross_val_score
#     scores = cross_val_score(clf, x1, y1, cv=k, scoring='roc_auc')
#     print(scores)
#     print("mean scores:", np.mean(scores))
#
#
# def plot_roc_curve(y_pred, y_pred_score):
#     fpr, tpr, thresholds = roc_curve(y_pred, y_pred_score)
#     roc_auc = auc(fpr, tpr)
#     # Plot ROC
#     plt.title('Receiver Operating Characteristic')
#     plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
#     plt.legend(loc='lower right')
#     plt.plot([0, 1], [0, 1], 'r--')
#     plt.xlim([-0.1, 1.0])
#     plt.ylim([-0.1, 1.01])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     # plt.show()
#     print("roc_auc:", roc_auc)
