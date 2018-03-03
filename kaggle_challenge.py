import pandas as pd
import csv as csv
import os
import pickle
from collections import Counter

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, KFold
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from metric_learn import LMNN, ITML_Supervised, LSML_Supervised, SDML_Supervised
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

train_data_path = './titanic/train.csv'
test_data_path = './titanic/test.csv'
save_path = './titanic/{}.bin'
output_path = './titanic/submit_data.csv'
alphabet_list = 'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z'.split(',')
columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Title', 'Cabin', 'Fare', 'FamilySize', 'IsAlone'] + alphabet_list
pp_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Title', 'Cabin', 'Fare', 'FamilySize', 'IsAlone'] + alphabet_list
pp2_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Title', 'Cabin', 'Fare', 'FamilySize', 'IsAlone']
objective_variable_name = 'Survived'
pp_objective_variable_name = 'Age'
pp2_objective_variable_name = 'Cabin'
id_name = 'PassengerId'

def data_frame_processing(df: 'DataFrame') -> 'DataFrame':
    df['Cabin'] = df['Cabin'].map(lambda x: str(x)[0] if x else x)
    # name process
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    # fitting Name to category distribution of alphabet
    prep_name_alphabet_category = df.Name.str \
        .extract('([A-Za-z\(\)]+)?,\s[A-Za-z]+\.([\(\)\sA-Za-z]+)', expand=False) \
        .apply(lambda x: '{0}{1}'.format(x[0], x[1]).lower(), axis=1) \
        .str.replace('[\s\(\)]', '').apply(lambda x: dict(Counter(x)))
    prep_name_alphabet_category = pd.DataFrame(list(prep_name_alphabet_category),
                                               index=prep_name_alphabet_category.index)
    prep_name_alphabet_category = prep_name_alphabet_category.fillna(0)
    df = pd.concat([df, prep_name_alphabet_category], axis=1)
    # family size process
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df['IsAlone'].loc[df['FamilySize'] > 1] = 0
    # fare process
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

    return df

def data_frame_make(train_file_path: str, test_file_path: str, columns: list) -> 'DataFrame':
    train_df = pd.read_csv(train_file_path, header=0)
    test_df = pd.read_csv(test_file_path, header=0)
    # TODO: deal with new comming category
    df = pd.concat([train_df, test_df])
    df = data_frame_processing(df)
    data = df[columns]
    return data, train_df

def data_read_pp2(train_file_path: str, test_file_path: str, columns: list, objective_variable_name: str) -> (list, list):
    

def data_read_pp(train_file_path: str, test_file_path: str, columns: list, objective_variable_name: str) -> (list, list):
    data, _ = data_frame_make(train_file_path, test_file_path, columns)
    data_dammies = pd.get_dummies(data)
    data_dammies = data_dammies.dropna(subset=['Age'])
    data_dammies = data_dammies.fillna(0)
    train_X = data_dammies.drop([objective_variable_name], axis=1).values
    train_y = data_dammies[[objective_variable_name]].values[:, 0]
    return train_X, train_y

def data_read(train_file_path: str, test_file_path: str, columns: list, objective_variable_name: str, regressor: dict) -> (
list, list, list):
    """
    data set read via pandas
    Args:
        - file_path: data set file path.
        - objective_variable_name: column name of objective value.
        - columns: read columns.
    Returns:
        - (train_X, train_y, test_X) : explanatory variables, objective value
    """
    data, train_df = data_frame_make(train_file_path, test_file_path, columns)
    train_len = len(train_df.index)

    # one-hot encoding
    data_dammies = pd.get_dummies(data)

    # regressor subst
    indexer = data_dammies['Age'].isnull()
    data_dammies = data_dammies.fillna(0)
    predicted = regressor.predict(data_dammies.drop(['Age', 'Survived'], axis=1).loc[indexer, :])
    data_dammies.loc[indexer, 'Age'] = predicted

    train_dammies = data_dammies[:train_len]
    test_dammies = data_dammies[train_len:]

    # vectorize
    train_X = train_dammies.drop([objective_variable_name], axis=1).values
    train_y = train_dammies[[objective_variable_name]].values[:, 0]
    test_X = test_dammies.drop([objective_variable_name], axis=1).values

    return train_X, train_y, test_X

def output_submit_data(file_path: str, test_file_path: str, test_y: list, id_name: str, objective_variable_name: str):
    test_df = pd.read_csv(test_file_path, header=0)
    ids = test_df[id_name].values

    with open(file_path, mode='w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([id_name, objective_variable_name])
        csv_writer.writerows(zip(ids, test_y))

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

class MLPipe:
    pipe = Pipeline([('scaling', StandardScaler()),
                     ('feature_selection',
                      SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')),
                     ('metric_learning', None),
                     ('classifier', SVC())])
    save_path = './titanic/pipe_{}.bin'
    feature_selection_param_grid = {
        'SVC': [
            {
                'scaling': [StandardScaler(), None],
                'metric_learning': [None,  LMNN()],
                'feature_selection': [SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median'), None],
                # 'feature_selection__estimator': [RandomForestClassifier(n_estimators=100, random_state=42), SVC(C=1000), KNeighborsClassifier()],
                'classifier': [SVC()],
                'classifier__kernel': ['rbf'],
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100]
            },
            {
                'scaling': [StandardScaler(), None],
                'metric_learning': [None,  LMNN()],
                'feature_selection': [SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median'), None],
                'classifier': [SVC()],
                'classifier__kernel': ['linear'],
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            }
        ],
        'rfc': [
            {
                #'scaling': [StandardScaler(), None],
                'scaling': [None],
                'metric_learning': [None,  LMNN()],
                'feature_selection': [SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median'), None],
                'classifier': [RandomForestClassifier()],
                'classifier__n_estimators': [10, 25, 50, 75, 100],
                'classifier__max_depth': [None, 5, 10, 25],
                'classifier__min_samples_split': [5, 10, 15]
            }
        ],
        'knn': [
            {
                'scaling': [StandardScaler(), MinMaxScaler(), None],
                'metric_learning': [None,  LMNN(), ITML_Supervised(num_constraints=200)],
                'feature_selection': [
                    SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median'),
                    None],
                'classifier': [KNeighborsClassifier()],
                'classifier__n_neighbors': [2, 3, 4, 5],
                'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree']
            }
        ],
        'dt': [
            {
                'scaling': [StandardScaler(), MinMaxScaler(), None],
                'metric_learning': [None],
                'feature_selection': [
                    SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median'),
                    None],
                'classifier': [DecisionTreeClassifier()],
                'classifier__criterion': ['gini', 'entropy'],
                'classifier__max_features': ['auto', 'sqrt', 'log2'],
                'classifier__max_depth': [None, 5, 10, 15]
            }
        ],
        'gbc': [
            {
                'scaling': [StandardScaler(), None],
                'metric_learning': [None, LMNN()],
                'feature_selection': [
                    SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median'),
                    None],
                'classifier': [GradientBoostingClassifier()],
                'classifier__loss': ['deviance'],
                'classifier__learning_rate': [0.1, 1, 10],
                'classifier__n_estimators': [10, 50, 100],
                'classifier__criterion': ['friedman_mse'],
                'classifier__max_features': ['auto'],
                'classifier__max_depth': [None, 2, 5, 10, 15, 25],
                'classifier__min_samples_split': [5, 10, 15]
            }
        ],
        'xgb': [
            {
                'scaling': [StandardScaler()],#, None],
                'metric_learning': [None],#, LMNN(), ITML_Supervised(num_constraints=200)],
                'feature_selection': [None],
                'classifier': [xgb.XGBClassifier()],
                'classifier__n_estimators': [500, 1000, 2000],
                'classifier__max_depth': [4, 6, 8, 10],
                'classifier__min_child_weight': [1, 2, 3],
                'classifier__gamma': [0.4, 0.6, 0.8, 0.9, 1],
                'classifier__subsample': [0.4, 0.6, 0.8, 1.0],
                'classifier__colsample_bytree': [0.4, 0.6, 0.8, 1.0]
            }
        ]
    }

    def __init__(self, model_name: str):
        print('model_name: %s' % model_name)
        self._model_name = model_name
        self._param_grid = MLPipe.feature_selection_param_grid[model_name]
        self._save_path = MLPipe.save_path.format(model_name)
        self._save_best_path = self._save_path + '-best'
        self._model = None
        self._pipe = MLPipe.pipe

    def fit_model(self, train_X: list, train_y: list):
        model = load_model(self._save_path)
        if not model:
            # create model, if not loading file
            grid_search = GridSearchCV(self._pipe, self._param_grid, cv=5, n_jobs=-1)
            grid_search.fit(train_X, train_y)
            save_model(self._save_path, grid_search)
            model = grid_search

        print('best score: {:.2f}'.format(model.best_score_))
        print('best estimator \n{}'.format(model.best_estimator_))
        self._model = model.best_estimator_

    def predict(self, test_X: list) -> list:
        test_y = self._model.predict(test_X).astype(int)
        return test_y

    def get_model(self) -> dict:
        return self._model

    def save_best_model(self):
        save_model(self._save_best_path, self._model)
    
    def load_best_model(self):
        self._model = load_model(self._save_best_path)

    def get_cv_failure_data(self, train_X: list, train_y: list):
        ret_index = np.array([])
        evaluate_model = self._model
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(train_X):
            evaluate_model.fit(train_X[train_index], train_y[train_index])
            evaluate_y = evaluate_model.predict(train_X[test_index])
            correct_eval_y = train_y[test_index]

            ret_index = np.concatenate((ret_index, np.array(test_index)[evaluate_y != train_y[test_index]]))

        return list(ret_index.astype(int))

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
    manifold_algorithms = {
        'tsne': TSNE,
        'Isomap': Isomap,
        'SE': SpectralEmbedding
    }

    def __init__(self, model_name: str, manifold_name):
        print('model_name: %s' % model_name)
        self._model_name = model_name
        self._algorithms = MachineLearning.algorithms[model_name]
        self._save_path = save_path.format(self._model_name)
        self._model = None
        self._select = None
        self._manifold = MachineLearning.manifold_algorithms[manifold_name]

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

    def fit_select_model(self, train_X: list, train_y: list):
        if self._model:
            self._select = SelectFromModel(self._model, threshold='median')
            self._select.fit(train_X, train_y)
            train_select_X = self._select.transform(train_X)
            print('train_X shape:{}'.format(train_X.shape))
            print('train_select_X shape:{}'.format(train_select_X.shape))
            self._model.fit(train_select_X, train_y)
            scores = cross_val_score(self._model, train_select_X, train_y, cv=5)
            print('cv-scores: {}'.format(scores))
            print('Average of cv-scores: {:.2f}'.format(scores.mean()))

    def embeding_vector(self, X: list, n_components: int, n_neighbors: int) -> list:
        X_embedded = self._manifold(n_components=n_components, n_neighbors=n_neighbors).fit_transform(X)
        return X_embedded

    def predict(self, test_X: list) -> list:
        if self._select:
            test_X = self._select.transform(test_X)
        test_y = self._model.predict(test_X).astype(int)
        return test_y

    def get_model(self) -> dict:
        return self._model

def main():
    train_X, train_y = data_read_pp(train_data_path, test_data_path, pp_columns, pp_objective_variable_name)
    regressor_ml = MachineLearning('SVR', 'tsne')
    regressor_ml.fit_model(train_X, train_y)
    regressor = regressor_ml.get_model()
    train_X, train_y, test_X = data_read(train_data_path, test_data_path, columns, objective_variable_name, regressor)

    ml = MLPipe('xgb')
    # ml = MachineLearning('knn', 'Isomap')

    ml.fit_model(train_X, train_y)
    # ml.fit_select_model(train_X, train_y)
    test_y = ml.predict(test_X)
    output_submit_data(output_path, test_data_path, test_y, id_name, objective_variable_name)

if __name__ == '__main__':
    main()