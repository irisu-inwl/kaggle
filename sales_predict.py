import csv as csv
import os
import pickle
from collections import Counter
from multiprocessing import Pool

import pandas as pd
import numpy as np

from fbprophet import Prophet
import matplotlib.pyplot as plt

directory_path = './sales_forecast/'

train_data_path = directory_path + 'sales_train_v2.csv'
test_data_path = directory_path + 'test.csv'
process_train_path = directory_path + 'sales_train_process.csv'
save_path = directory_path + '{}.bin'
output_path = directory_path + 'submit_data.csv'
len_month_path = directory_path + 'test_len_month.csv'
objective_variable_name = 'item_cnt_month'
id_name = 'ID'

def data_frame_make(train_data_path: str, test_data_path: str) -> 'DataFrame':
    train_df = pd.read_csv(train_data_path, header=0)
    test_df = pd.read_csv(test_data_path, header=0)
    # TODO: deal with new comming category
    return train_df, test_df

def data_read(train_data_path: str, test_data_path: str):
    train_df,test_df = data_frame_make(train_data_path, test_data_path)
    return train_df,test_df

def get_average(df: 'DataFrame') -> int:
    df_date_t = df.copy()
    df_date_t['ds'] = pd.to_datetime(df_date_t['ds'], format='%Y-%m-%d')
    df_date_t = df_date_t.groupby(pd.Grouper(key='ds', freq='M')).sum()
    return df_date_t.mean()[0]

def output_submit_data(file_path: str, test_file_path: str, test_y: list, id_name: str, objective_variable_name: str):
    test_df = pd.read_csv(test_file_path, header=0)
    ids = test_df[id_name].values

    with open(file_path, mode='w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([id_name, objective_variable_name])
        csv_writer.writerows(zip(ids, test_y))

def output_submit_df(file_path: str, df: 'DataFrame'):
    df.to_csv(file_path, index=False)

def subproc_prophet(df: 'DataFrame', shop: str, item: str, submit_id):
    make_dict = lambda score: {id_name: submit_id, objective_variable_name: score}

    df_date_cnt = df.loc[(df.shop_id == shop) & (df.item_id == item), ['date','item_cnt_day']].sort_values(by='date')
    df_date_cnt = df_date_cnt.rename(columns={'date': 'ds', 'item_cnt_day': 'y'})

    if len(df_date_cnt) == 0:
        return make_dict(0)
    if df_date_cnt.ds.str.extract('([0-9]+)?-([0-9]+)?')[-1:][0].values[0] != '2015':
        return make_dict(0)
    if int(df_date_cnt.ds.str.extract('([0-9]+)?-([0-9]+)?')[-1:][1].values[0]) <= 8:
        return make_dict(0)
    if len(df_date_cnt) <= 5:
        return make_dict(get_average(df_date_cnt))

    # define saturation
    df_date_cnt['cap'] = 20
    df_date_cnt['floor'] = 0

    # predict via prophet
    get_index = 11 - int(df_date_cnt.ds.str.extract('([0-9]+)?-([0-9]+)?')[-1:][1].values[0])
    model = Prophet()
    model.fit(df_date_cnt)
    future_data = model.make_future_dataframe(periods=get_index+1, freq = 'm')
    future_data['cap'] = 20
    future_data['floor'] = 0
    forecast_data = model.predict(future_data)
    forecast_month = forecast_data.yhat.tail(1).values[0]
    return make_dict(forecast_month)

def data_predict(train_df, test_df, len_month_df):
    # predict_series = []
    with Pool(10) as p:
        predict_series = p.starmap(subproc_prophet, zip([train_df]*len(test_df.ID), test_df.shop_id, test_df.item_id, test_df.ID))
#    for shop, item, submit_id in zip(test_df.shop_id, test_df.item_id, test_df.ID):
#        df_date_cnt = train_df.loc[(train_df.shop_id == shop) & (train_df.item_id == item), ['date','item_cnt_day']].sort_values(by='date')
#        df_date_cnt = df_date_cnt.rename(columns={'date': 'ds', 'item_cnt_day': 'y'})
#        if len(df_date_cnt) == 0:
#            predict_series.append(0)
#            continue
#        if df_date_cnt.ds.str.extract('([0-9]+)?-([0-9]+)?')[-1:][0].values[0] != '2015':
#            predict_series.append(0)
#            continue
#        if int(df_date_cnt.ds.str.extract('([0-9]+)?-([0-9]+)?')[-1:][1].values[0]) <= 8:
#            predict_series.append(0)
#            continue
#        if len(df_date_cnt) <= 5:
#            predict_series.append(get_average(df_date_cnt))
#            continue

        # predict via prophet        
#        get_index = 11 - int(df_date_cnt.ds.str.extract('([0-9]+)?-([0-9]+)?')[-1:][1].values[0])
#        model = Prophet()
#        model.fit(df_date_cnt)
#        future_data = model.make_future_dataframe(periods=get_index+1, freq = 'm')
#        forecast_data = model.predict(future_data)
#        predict_series.append(forecast_data.yhat.tail(1).values[0])

    return predict_series

def main():
    train_df, test_df = data_read(process_train_path, test_data_path)
    len_month_df = pd.read_csv(len_month_path, header=0)
    predict_result = data_predict(train_df,test_df, len_month_df)
    output_df = pd.DataFrame(predict_result)
    output_submit_df(output_path, output_df)

if __name__ == '__main__':
    main()
