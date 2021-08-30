# Copyright (c) 2021 Dai HBG

"""
该代码定义的类用于调用聚宽的接口查询数据
This code is used for calling the interface of Join Quant to query data

v1.0
2021-08-30
-- 暂时不支持分钟数据，因为聚宽的数据查询有次数限制
-- get_pv_data方法的参数data_type传入一个列表，表示需要的数据
"""

import numpy as np
import os
import pandas as pd
import pickle
import jqdatasdk
from jqdatasdk import *
import datetime


class DataLoader:
    def __init__(self, user_id, password, data_path='F:/Documents/AutoFactoryData'):
        self.data_path = data_path
        self.user_id = user_id
        self.password = password
        jqdatasdk.auth(self.user_id, self.password)  # 登录聚宽

    def get_pv_data(self, date, data_type=None):  # 获得日频量价关系数据
        if data_type is None:  # 参数默认值不要是可变的，否则可能出错
            data_type = ['stock_daily']

        if 'stock_daily' in data_type:
            all_stocks = list(get_all_securities(types=['stock'], date=date).index)
            lst = os.listdir('{}/StockDailyData'.format(self.data_path))
            if date not in lst:
                os.makedirs('{}/StockDailyData/{}'.format(self.data_path, date))
            day = date.split('-')
            end_date = str(datetime.date(int(day[0]), int(day[1]), int(day[2])).timedelta(days=1))
            stock_data = get_price(all_stocks, frequency='daily',
                                   field=['open', 'close', 'low', 'high', 'volume', 'money', 'pre_close', 'factor'],
                                   start_date=date, end_date=end_date)
            with open('{}/StockDailyData/{}/stock_{}.pkl'.format(self.data_path, date, date), 'wb') as f:
                pickle.dump(stock_data, f)
        if 'index_daily' in data_type:
            all_indexes = get_all_securities(types=['index'], date=date)
            with open('{}/StockDailyData/{}/index_{}.pkl'.format(self.data_path, date, date), 'wb') as f:
                pickle.dump(all_indexes, f)

        if 'minute' in data_type:
            lst = os.listdir('{}/StockIntraDayData'.format(self.data_path))
            if date not in lst:
                os.makedirs('{}/StockIntraDayData/{}'.format(self.data_path, date))
                all_stocks = list(get_all_securities(types=['stock'], date=date).index)
                for stock in all_stocks:  # 剔除创业板股票，避免超出查询限制
                    if stock[:3] == '003':
                        continue
                    day = date.split('-')
                    end_date = str(datetime.date(int(day[0]), int(day[1]), int(day[2])).timedelta(days=1))
                    intra_day_data = get_price(stock, frequency='minute',
                                               field=['open', 'close', 'low', 'high', 'volume', 'money', 'pre_close',
                                                      'factor'],
                                               start_date=date, end_date=end_date)
                    with open('{}/StockIntraDayData/{}/{}.pkl'.format(self.data_path, date, stock), 'wb') as f:
                        pickle.dump(intra_day_data, f)
