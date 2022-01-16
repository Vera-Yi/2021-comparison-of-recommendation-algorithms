from surprise.model_selection import KFold
from surprise import SVD,KNNBaseline
from surprise import accuracy
from surprise import Dataset
import numpy as np
import pandas as pd

k1 = 4
data = Dataset.load_builtin('ml-100k')
algo = KNNBaseline()

kf = KFold(n_splits=k1)
test_result = []
for trainset,testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    test_result.append(accuracy.mae(predictions,verbose=True))
print("{0}折交叉验证的结果是{1}".format(k1,np.mean(test_result)))
