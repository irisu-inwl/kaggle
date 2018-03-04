import pandas as pd
import csv as csv
import os
import pickle
from collections import Counter

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
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from ethnicolr import pred_census_ln, pred_wiki_ln

train_data_path = './titanic/train.csv'
test_data_path = './titanic/test.csv'
save_path = './titanic/{}.bin'
output_path = './titanic/submit_data.csv'
alphabet_list = 'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z'.split(',')
columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Title', 'Cabin', 'Fare', 'FamilySize', 'IsAlone', 'NameLength', 'TicketPre', 'TicketNum', 'Race', 'Country'] + alphabet_list
pp_columns_age = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Title', 'Fare', 'FamilySize', 'IsAlone', 'NameLength', 'Race', 'Country'] + alphabet_list
pp_columns_cabin = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Title', 'Cabin', 'Fare', 'FamilySize', 'IsAlone', 'TicketPre', 'TicketNum', 'Race']
categorical_column = [columns.index(name) - 1 for name in ['Sex', 'Embarked', 'Title', 'Cabin', 'TicketPre']]
objective_variable_name = 'Survived'
pp_objective_variable_name_age = 'Age'
pp_objective_variable_name_cabin = 'Cabin'
id_name = 'PassengerId'

def data_frame_processing(df: 'DataFrame') -> 'DataFrame':
    df['Cabin'] = df['Cabin'].map(lambda x: str(x)[0] if x else x)
    # name process
    df['Title'] = df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # fitting Name to category distribution of alphabet
    prep_name_alphabet_category = df.Name.str \
        .extract(r'([A-Za-z\(\)]+)?,\s[A-Za-z]+\.([\(\)\sA-Za-z]+)', expand=False) \
        .apply(lambda x: '{0}{1}'.format(x[0], x[1]).lower(), axis=1) \
        .str.replace(r'[\s\(\)]', '').apply(lambda x: dict(Counter(x)))
    prep_name_alphabet_category = pd.DataFrame(list(prep_name_alphabet_category),
                                               index=prep_name_alphabet_category.index)
    prep_name_alphabet_category = prep_name_alphabet_category.fillna(0)
    df = pd.concat([df, prep_name_alphabet_category], axis=1)

    # name length process
    df['NameLength'] = df.Name.apply(lambda x: len(x)).astype(int)

    # family size process
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df['IsAlone'].loc[df['FamilySize'] > 1] = 0

    # fare process
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    # df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    # df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    # df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    # df.loc[df['Fare'] > 31, 'Fare'] = 3
    # df['Fare'] = df['Fare'].astype(int)
    df['Fare'] = df['Fare']+1
    df['Fare'] = df['Fare'].apply(np.log)
    df['Fare'] = df['Fare']

    # Ticket Pre and Number process
    TkPre_TkNum = df.Ticket.str.extract(r'([A-Za-z/0-9\.]+)?\s[A-Za-z\s]*([0-9]+)?', expand=False)
    TkNum = df.Ticket.str.extract(r'([0-9]+)?', expand=False)
    TkPre_TkNum.loc[TkNum.isnull() == False,1] = TkNum[TkNum.isnull() == False]
    TkPre_TkNum.loc[:, 0] = TkPre_TkNum.loc[:, 0].str.replace(r'\.', '')
    TkPre_TkNum.loc[:, 0][df.Ticket == 'LINE'] = 'LINE'
    TkPre_TkNum.loc[:, 0] = TkPre_TkNum.loc[:, 0].fillna('n')
    df['TicketPre'] = TkPre_TkNum.loc[:, 0]
    TkPre_TkNum.loc[:, 1] = TkPre_TkNum.loc[:, 1].fillna(0).astype(float)
    df['TicketNum'] = TkPre_TkNum.loc[:, 1]

    # country and race predict
    name_series = df.Name.str.extract(r'([A-Za-z\(\)]+)?,\s[A-Za-z]+\.\s[\(]*([a-zA-Z]+)?')
    name_df = pd.DataFrame(name_series)
    name_df = name_df.rename(columns={0:'last_name',1:'first_name'})
    census_df = pred_census_ln(name_df, 'first_name')
    df['Race'] = census_df['race']
    wiki_ln_df = pred_wiki_ln(name_df, 'last_name')
    df['Country'] = wiki_ln_df.race.str.extract(r'[a-zA-Z,]*[a-zA-Z]+,([a-zA-Z]+)?')

    return df

def data_frame_make(train_file_path: str, test_file_path: str) -> 'DataFrame':
    train_df = pd.read_csv(train_file_path, header=0)
    test_df = pd.read_csv(test_file_path, header=0)
    # TODO: deal with new comming category
    df = pd.concat([train_df, test_df])
    return df, train_df

def preprocessing(df_source: 'DataFrame', columns: list, objective_variable_name: str, ml_name: str) -> 'DataFrame':
    # for preprocessing supervised learning
    data = df_source.copy()

    if objective_variable_name == 'Cabin':
        data['Age'] = data['Age'].fillna(data['Age'].median())

    indexer_null = data[objective_variable_name].isnull()
    indexer_not_null = (data[objective_variable_name].isnull() == False)

    data = data_frame_processing(data)
    data = data[columns]

    # 削る前にtrain_yを入れておく
    train_y = data[indexer_not_null][objective_variable_name].values

    # drop, onehot encoding, fill missing_value
    df_drop = data.drop([objective_variable_name, 'Survived'], axis=1)
    df_drop = pd.get_dummies(df_drop)
    df_drop = df_drop.fillna(0)

    train_X = df_drop[indexer_not_null].values

    # fitting classifier

    classifier_ml = MLPipe(ml_name, '_{0}_pre'.format(len(columns)))
    classifier_ml.fit_model(train_X, train_y)
    classifier = classifier_ml.get_model()
    
    # predict classifier
    clf_X = df_drop[indexer_null].values
    predicted = classifier.predict(clf_X)
    df_source.loc[indexer_null, objective_variable_name] = predicted

    return df_source

def get_categorical_dict(df:'DataFrame' ,columns: list, categorical_column: list) -> (dict, 'DataFrame'):
    categorical_names = {}
    for feature in categorical_column:
        le = LabelEncoder()
        le.fit(df[columns[feature+1]].astype(str))
        df[columns[feature+1]] = le.transform(df[columns[feature+1]].astype(str))
        categorical_names[feature] = le.classes_
    return categorical_names, df

def data_read(df: 'DataFrame', train_df: 'DataFrame', columns: list, objective_variable_name: str) -> (
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
    train_len = len(train_df.index)    

    # cabin classifier
    #  preleminaire

    data = data_frame_processing(df)
    data = data[columns]
    
    # one-hot encoding
    data_dammies = pd.get_dummies(data)

    train_dammies = data_dammies[:train_len]
    test_dammies = data_dammies[train_len:]

    data_dammies = data_dammies.fillna(0)

    # vectorize
    train_X = train_dammies.drop([objective_variable_name], axis=1).values
    train_y = train_dammies[[objective_variable_name]].values[:, 0]
    test_X = test_dammies.drop([objective_variable_name], axis=1).values

    return train_X, train_y, test_X

def data_read_for_lime(df: 'DataFrame', train_df: 'DataFrame', columns: list, categorical_column: list, objective_variable_name: str) -> (dict, list):
    train_len = len(train_df.index)

    categorical_names, data = get_categorical_dict(df, columns, categorical_column)
    
    train_data = data[:train_len]

    train_X = train_data.drop([objective_variable_name], axis=1).values
    return categorical_names, train_X

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
        ]
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

def main():
    # preprocessing cabin
    # train_X, train_y = data_read_pp2(train_data_path, test_data_path, pp2_columns, pp2_objective_variable_name)

    # data read
    df, train_df = data_frame_make(train_data_path, test_data_path)

    # data preprocessing
    print('#########################')
    print('missing_value of Cabin filling')
    print('#########################')
    df = preprocessing(df, pp_columns_cabin, pp_objective_variable_name_cabin, 'multi_xgb')

    print('#########################')
    print('missing_value of Age filling')
    print('#########################')
    df = preprocessing(df, pp_columns_age, pp_objective_variable_name_age, 'SVR')

    # print(df.isnull().sum())

    # main predict
    train_X, train_y, test_X = data_read(df, train_df, columns, objective_variable_name)

    ml = MLPipe('xgb', '_{0}_main'.format(len(columns)))

    ml.fit_model(train_X, train_y)
    test_y = ml.predict(test_X)
    output_submit_data(output_path, test_data_path, test_y, id_name, objective_variable_name)

if __name__ == '__main__':
    main()