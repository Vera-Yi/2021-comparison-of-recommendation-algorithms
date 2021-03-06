#-*-coding:utf-8-*-

import csv
import sys

import pandas as pd

sys.path.insert(0, '..')
from classic_recommender.KNN_user_movies import Personal_KNN_recommender
from classic_recommender.Personal_SVD import Personal_SVD_recommender
import datetime
# 首先用KNN对输入的用户进行相似度匹配，然后挑选出最接近的10个其他用户
# 之后对于选出的电影，根据SVD计算用户对电影的模拟评分来进行排序
starttime = datetime.datetime.now()
class KNN_SVD_ensemble:
    def __init__(self):
        self.user = Personal_KNN_recommender()
        self.movie = Personal_SVD_recommender()
        self.testings = pd.read_csv('../data/personal/movie_test.csv')
        self.userid = []
        for i in range(len(self.testings['userId'])):
            if not self.testings['userId'][i] in self.userid:
                self.userid.append(self.testings['userId'][i])


    def recommend(self, usrID):
        _, first_ids = self.user.recommend(usrID, 50)
        # print(first_ids)
        second_ids, movie_id = self.movie.recommend(usrID, first_ids, 10)
        # print(second_ids)
        return movie_id

    def test(self, num):
        result = []
        for user in self.userid:
            print(user)
            ids = self.recommend(user)
            print(ids)
            result.append(ids)

        with open("./result.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['userId', 'result'])
            for i, row in enumerate(result):
                writer.writerow([self.userid[i], row])


test = KNN_SVD_ensemble()
# test.recommend(2)
test.test(10)
endtime = datetime.datetime.now()
print(endtime - starttime)