from surprise import KNNWithMeans, accuracy, KNNBasic, KNNWithZScore, KNNBaseline
from surprise import Dataset, Reader
from surprise.model_selection import KFold

# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('knn_cf/ratings.csv', reader=reader)

#KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
algo = KNNWithMeans(k=50, sim_options={'user_based': False, 'verbose': 'True'})

bsl_options = {'method': 'als',
               'n_epochs': 20,
               }
sim_options = {'name': 'pearson_baseline'}
algo = KNNBasic(bsl_options=bsl_options, sim_options=sim_options)

algo = KNNWithZScore(k=50, sim_options={'user_based': False, 'verbose': 'True'})

algo = KNNBaseline(k=50, sim_options={'user_based': False, 'verbose': 'True'})

kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RMSE
    accuracy.rmse(predictions, verbose=True)