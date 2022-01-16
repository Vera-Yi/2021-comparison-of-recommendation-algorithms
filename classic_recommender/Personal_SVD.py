#-*-coding:utf-8-*-
# implement the Collaborative Filtering for recommending
import datetime
import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
starttime = datetime.datetime.now()
class Personal_SVD_recommender:
    def __init__(self):
        self.reader = Reader()
        self.ratings = pd.read_csv('../data/personal/movie_train.csv')
        data = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], self.reader)
        # data.split(n_folds=5)
        self.svd = SVD(n_epochs=10, n_factors=100, biased=True,lr_all=0.007, reg_all=0.2, verbose=False)
        #cross_validate(self.svd, data, measures=['RMSE', 'MAE'])
        trainset = data.build_full_trainset()
        self.svd.fit(trainset)

        self.index = pd.read_csv('../data/personal/movies.csv')

    def rating(self, usrID, movieID):
        rate = self.svd.predict(usrID, movieID)
        return rate[3]
    def sample_movies(self, n):
        pass

    # 主要用于对给定的列表，进行用户模拟评分，之后进一步挑选，属于二阶步骤
    def recommend(self, usrID, movies, num=30):
        dic = {}
        for i in movies:
            dic[i] = self.rating(usrID, i)
        # print('dic', dic)
        result = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        result = result[:num]
        # print('result', result)
        movie = []
        rates = []
        ids = []
        for i in result:
            # print(i)
            movie.append(self.index[self.index.movieId==i[0]]['title'])
            rates.append(i[1])
            ids.append(i[0])
        return movie, ids


test = Personal_SVD_recommender()
print(test.recommend(2, [1,2,3,4,5,6,7,8,9,10,11,12,13,14]))
endtime = datetime.datetime.now()
print(endtime - starttime)