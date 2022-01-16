from surprise import SVD
from surprise import Dataset
from surprise.model_selection import GridSearchCV


data = Dataset.load_builtin('ml-100k')

para_grid = {'n_epochs':[5],
             'n_factors':[100],
             'lr_all':[0.07],
             'reg_all':[0.2]}
gs = GridSearchCV(SVD, para_grid, measures=['rmse','mae','fcp'],cv=5)

gs.fit(data)

#获得最好得分
print(gs.best_score['fcp'])

