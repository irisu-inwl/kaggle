import pandas as pd
import csv as csv
import os
import pickle
from collections import Counter

from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

save_path = './titanic/{}.bin'

def save_model(file_path: str, data: dict):
    with open(file_path, mode='wb') as file:
        pickle.dump(data, file)

def load_model(file_path: str) -> dict:
    ret = None
    if not os.path.exists(file_path):
        return ret
    if not os.path.isfile(file_path):
        return ret

    with open(file_path, mode='rb') as file:
        ret = pickle.load(file)

    return ret

class MachineLearning:
    algorithms = {
        'SVC': {
            'param_grid': [  # 'linear', 'rbf', 'poly', 'sigmoid'
                {
                    'kernel': ['linear'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100]
                },
                {
                    'kernel': ['rbf'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
                },
#                {
#                    'kernel': ['sigmoid'],
#                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
#                    'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
#                    'coef0': [0.001, 0.01, 0.1, 1, 10, 100]
#                }
            ],
            'method': SVC()
        },
        'rfc': {
            'param_grid': {
                'n_estimators': [10, 25, 50, 75, 100],
                'criterion': ['gini', 'entropy'],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': [None, 2, 5, 10, 15, 25]
            },
            'method': RandomForestClassifier()
        },
        'gbc': {
            'param_grid': {
                'loss': ['deviance', 'exponential'],
                'learning_rate': [0.1, 1, 10],
                'n_estimators': [10, 50, 100],
                'criterion': ['friedman_mse', 'mse'],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': [None, 2, 5, 10, 15, 25]
            },
            'method': GradientBoostingClassifier()
        },
        'knn': {
            'param_grid': {
                'n_neighbors': [2, 3, 4, 5],
                'algorithm': ['auto', 'ball_tree', 'kd_tree']
            },
            'method': KNeighborsClassifier()
        },
        'SVR': {
            'param_grid': [  # 'linear', 'rbf', 'poly', 'sigmoid'
                {
                    'kernel': ['linear'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100]
                },
                {
                    'kernel': ['rbf'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
                },
#                {
#                    'kernel': ['sigmoid'],
#                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
#                    'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
#                    'coef0': [0.001, 0.01, 0.1, 1, 10, 100]
#                }
            ],
            'method': SVR()
        }
    }

    def __init__(self, model_name: str):
        print('model_name: %s' % model_name)
        self._model_name = model_name
        self._algorithms = MachineLearning.algorithms[model_name]
        self._save_path = save_path.format(self._model_name)
        self._model = None

    def fit_model(self, train_X: list, train_y: list):
        model = load_model(self._save_path)
        if not model:
            # create model, if not loading file
            grid_search = GridSearchCV(self._algorithms['method'], self._algorithms['param_grid'], cv=5, n_jobs=-1)
            grid_search.fit(train_X, train_y)
            save_model(self._save_path, grid_search)
            model = grid_search

        print('best score: {:.2f}'.format(model.best_score_))
        # print('best estimator \n{}'.format(model.best_estimator_))
        self._model = model.best_estimator_

    def predict(self, test_X: list) -> list:
        test_y = self._model.predict(test_X).astype(int)
        return test_y

    def get_model(self) -> dict:
        return self._model