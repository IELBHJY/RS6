from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly
from surprise import accuracy
from surprise.model_selection import KFold
import pandas as pd
import numpy as np

pd.options.display.max_columns = None
pd.options.display.max_rows = None

def process_df(df):
    bool_contain_item = df[0].str.contains(':')
    df.loc[df[bool_contain_item].index,'movie_id'] = df.loc[df[bool_contain_item].index,0].str.slice(0,-1)
    df['movie_id'].fillna(method='ffill',inplace=True)
    df.drop(df[bool_contain_item].index,axis=0,inplace=True)
    split_tmp_df=df[0].str.split(',',expand=True).rename({0:'user_id',1:'rating',2:'timestamp'},axis=1)
    df = pd.concat([df.drop([0],axis=1),split_tmp_df],axis=1)
    users = df['user_id']
    df.drop(labels=['user_id'], axis=1, inplace=True)
    df.insert(0, 'user_id', users)
    df.to_csv('first_file.csv', header=True, index=False)
    return df

def process_probe(df):
    bool_contain_item = df[0].str.contains(':')
    df.loc[df[bool_contain_item].index,'movie_id'] = df.loc[df[bool_contain_item].index,0].str.slice(0,-1)
    df['movie_id'].fillna(method='ffill',inplace=True)
    df.drop(df[bool_contain_item].index,axis=0,inplace=True)
    df.columns=['user_id','movie_id']
    df.to_csv('probe.csv',header=True,index=False)
    return df

#读取训练数据
rate_data = pd.read_table('combined_data_1.txt',sep='/t',header=None)
#数据处理
data = process_df(rate_data)

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
suprise_data = Dataset.load_from_file('../Kaggle/netflix-prize-data/first_file.csv', reader=reader)
train_set = suprise_data.build_full_trainset()

# ALS优化
bsl_options = {'method': 'als','n_epochs': 50,'reg_u': 12,'reg_i': 5}
# SGD优化
#bsl_options = {'method': 'sgd','n_epochs': 5}
algo = BaselineOnly(bsl_options=bsl_options)
#algo = BaselineOnly()
#algo = NormalPredictor()

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(suprise_data):
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RMSE
    accuracy.rmse(predictions, verbose=True)

#读取需要预测数据，并处理
probe = pd.read_table('probe.txt',sep='/t',header=None)
processed_probe = process_probe(probe)

#由于训练数据读取一部分，需要筛选出在train中出现的user_id
pre = pd.merge(data, processed_probe,how='inner',on=['user_id','movie_id'])
print('start predict')
############最终结果0.989714596450271################
count = 0
error=0
for user,movie in zip(pre['user_id'],pre['movie_id']):
    rui = pre[(pre.user_id == str(user)) & (pre.movie_id == str(movie))].loc[:,'rating']
    count += 1
    rui_value = int(rui.iloc[0])
    prediction = algo.predict(str(user), str(movie), r_ui=rui_value, verbose=True)
    error += np.square(prediction[3] - rui_value)
print("RMSE:{}".format(np.sqrt(error/count)))
