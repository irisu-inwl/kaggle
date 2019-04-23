import pandas as pd
import csv as csv

from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from ethnicolr import pred_census_ln, pred_wiki_ln

from ml_pipeline import MLPipe

from catboost import CatBoostClassifier, Pool, cv

train_data_path = './titanic/train.csv'
test_data_path = './titanic/test.csv'
save_path = './titanic/{}.bin'
output_path = './titanic/submit_data.csv'
alphabet_list = 'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z'.split(',')
grand_column = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Title', 'Cabin', 'Fare', 'FamilySize', 'IsAlone', 'NameLength', 'TicketPre', 'TicketNum', 'Race', 'Country'] + alphabet_list
columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Title', 'Cabin', 'Fare', 'FamilySize', 'IsAlone', 'NameLength', 'TicketPre', 'Race', 'Country']
pp_columns_age = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Title', 'Fare', 'FamilySize', 'IsAlone', 'NameLength', 'Race', 'Country'] + alphabet_list
pp_columns_cabin = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Title', 'Cabin', 'Fare', 'FamilySize', 'IsAlone', 'TicketPre', 'TicketNum', 'Race']
categorical_column_name = ['Sex', 'Embarked', 'Title', 'Cabin', 'TicketPre', 'Race', 'Country'] 
categorical_column = [columns.index(name) - 1 for name in categorical_column_name if name in columns]
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

from sklearn.metrics import accuracy_score

def predict_cat(df: 'DataFrame', train_df: 'DataFrame', columns: list, objective_variable_name: str) -> list:
    train_len = len(train_df.index)    

    data = data_frame_processing(df)
    data = data[columns]
    data = data.fillna(0)

    train_df = data[:train_len]
    test_df = data[train_len:]

    X = train_df.drop(objective_variable_name, axis=1)
    y = train_df[objective_variable_name]
    test_X = test_df.drop(objective_variable_name, axis=1)

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8, random_state=1234)

    model = CatBoostClassifier(iterations=5000, calc_feature_importance = True, use_best_model=True, eval_metric = 'Accuracy' )
    model.fit(X_train, y_train, cat_features=categorical_column, eval_set=(X_validation, y_validation))

    y_val_pred = model.predict(X_validation)
    print('accuracy : %.2f' % accuracy_score(y_val_pred, y_validation))
    # model.fit(X, y, cat_features=categorical_column)

    # cv_data = cv(model.get_params(), Pool(X, label=y, cat_features=categorical_column))

#    print('Best validation accuracy score: {:.2f}±{:.2f} on step {}'.format( \
#        np.max(cv_data['Accuracy_test_avg']), \
#        cv_data['Accuracy_test_stddev'][np.argmax(cv_data['Accuracy_test_avg'])], \
#        np.argmax(cv_data['Accuracy_test_avg'])) \
#    )

    test_y = model.predict(test_X)
    test_y = test_y.astype(int)

    return test_y


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
#    train_X, train_y, test_X = data_read(df, train_df, columns, objective_variable_name)
#    ml = MLPipe('catboost', '_{0}_main'.format(len(columns)))
#    ml.fit_model(train_X, train_y)
#    test_y = ml.predict(test_X)
    test_y = predict_cat(df, train_df, columns, objective_variable_name)
    output_submit_data(output_path, test_data_path, test_y, id_name, objective_variable_name)

if __name__ == '__main__':
    main()