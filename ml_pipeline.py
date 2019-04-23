import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
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
from catboost import CatBoostClassifier

from util import save_model, load_model

columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Title', 'Cabin', 'Fare', 'FamilySize', 'IsAlone', 'NameLength', 'TicketPre', 'Race', 'Country']
categorical_column_name = ['Sex', 'Embarked', 'Title', 'Cabin', 'TicketPre', 'Race', 'Country'] 
categorical_column = [columns.index(name) - 1 for name in categorical_column_name if name in columns]

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
                'classifier__C': [0.001, 0.01, 0.1, 1],
                'classifier__gamma': [0.001, 0.01, 0.1, 1]
            },
            {
                'scaling': [StandardScaler(), None],
                'metric_learning': [None,  LMNN()],
                'feature_selection': [SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median'), None],
                'classifier': [SVC()],
                'classifier__kernel': ['linear'],
                'classifier__C': [0.001, 0.01, 0.1, 1],
            }
        ],
        'multiSVC': [
            {
                'scaling': [StandardScaler(), None],
                'metric_learning': [None],
                'feature_selection': [SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median'), None],
                # 'feature_selection__estimator': [RandomForestClassifier(n_estimators=100, random_state=42), SVC(C=1000), KNeighborsClassifier()],
                'classifier': [SVC()],
                'classifier__kernel': ['rbf'],
                'classifier__C': [0.001, 0.01, 0.1, 1],
                'classifier__gamma': [0.001, 0.01, 0.1, 1]
            },
            {
                'scaling': [StandardScaler(), None],
                'metric_learning': [None],
                'feature_selection': [SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median'), None],
                'classifier': [SVC()],
                'classifier__kernel': ['linear'],
                'classifier__C': [0.001, 0.01, 0.1, 1, 10],
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
        'multi_rfc': [
            {
                #'scaling': [StandardScaler(), None],
                'scaling': [None],
                'metric_learning': [None],
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
                'scaling': [StandardScaler()],
                'metric_learning': [None],#, LMNN(), ITML_Supervised(num_constraints=200)],
                'feature_selection': [None],
                'classifier': [xgb.XGBClassifier()],
                'classifier__n_estimators': [500, 1000, 2000],
                'classifier__max_depth': [3, 4, 5, 6],
                'classifier__min_child_weight': [1, 2],
                'classifier__gamma': [0.35, 0.2, 0.5, 0.6, 0.8],
                'classifier__subsample': [0.55, 0.35, 0.8, 1.0],
                'classifier__colsample_bytree': [0.2, 0.4, 0.6, 0.8]
            }
        ],
        'multi_xgb': [
            {
                'scaling': [StandardScaler()],
                'metric_learning': [None],#, LMNN(), ITML_Supervised(num_constraints=200)],
                'feature_selection': [SelectFromModel(xgb.XGBClassifier(n_estimators=2000), threshold='median')],
                'classifier': [xgb.XGBClassifier()],
                'classifier__n_estimators': [1000, 2000],
                'classifier__max_depth': [3, 4],
                'classifier__min_child_weight': [1],
                'classifier__gamma': [0.35, 0.2, 0.5],
                'classifier__subsample': [0.55, 0.35, 0.75],
                'classifier__colsample_bytree': [0.2, 0.4, 0.6],
                'classifier__objective': ['multi:softprob']
            }
        ],
        'SVR': [
            {
                'scaling': [StandardScaler(), None],
                'metric_learning': [None],
                'feature_selection': [SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold='median'), None],
                'classifier': [SVR()],
                'classifier__kernel': ['rbf'],
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100]
            }
        ],
        'catboost': [
            {
                'scaling': [None],
                'metric_learning': [None],#, LMNN(), ITML_Supervised(num_constraints=200)],
                'feature_selection': [None],
                'classifier': [CatBoostClassifier(calc_feature_importance = True, use_best_model=True, eval_metric = 'Accuracy')],
                'classifier__iterations': [1000, 2000, 3000, 5000],
                'classifier__alpha': [0.4, 0.5, 0.6],
                'classifier__border': [0.4, 0.5, 0.6]
            }
        ],
    }

    def __init__(self, model_name: str, prefix: str = ''):
        print('model_name: %s' % model_name)
        self._model_name = model_name
        self._param_grid = MLPipe.feature_selection_param_grid[model_name]
        self._save_path = MLPipe.save_path.format(model_name+prefix)
        self._save_best_path = self._save_path + '-best'
        self._model = None
        self._pipe = MLPipe.pipe

    def fit_model(self, train_X: list, train_y: list):
        model = load_model(self._save_path)
        if not model:
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