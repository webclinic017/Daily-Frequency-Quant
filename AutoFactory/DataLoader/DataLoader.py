# Copyright (c) 2021 Dai HBG

"""
该代码定义的类用于调用聚宽的接口查询数据
This code is used for calling the interface of Join Quant to query data
"""

import numpy as np
import os
import pandas as pd
import pickle
import jqdatasdk
from jqdatasdk import *


class DataLoader:
    def __init__(self, user_id, password, data_path='F:/Documents/AutoFactoryData'):
        self.data_path = data_path
        self.user_id = user_id
        self.password = password
        jqdatasdk.auth(self.user_id, self.password)  # 登录聚宽

    def get_pv_data(self, date, data_type='daily'):  # 获得日频量价关系数据
        all_stocks = list(get_all_securities(types=['stock'], date=None).index)
        if data_type == 'daily':
            all_indexes = get_all_securities(types=['stock'], date=None)
            lst = os.listdir('{}/StockDailyData'.format(self.data_path))
            if date not in lst:
                os.makedirs('{}/StockDailyData/{}'.format(self.data_path, date))
            with open('{}/StockDailyData/{}/index_{}.pkl'.format(self.data_path, date, date), 'wb') as f:
                pickle.dump(all_indexes, f)
            stock_data = get_price(all_stocks,frequency='daily',date=date)
            with open('{}/StockDailyData/{}/stock_{}.pkl'.format(self.data_path, date, date), 'wb') as f:
                pickle.dump(stock_data, f)
        elif data_type == 'minute':
            lst = os.listdir('{}/StockIntraDayData'.format(self.data_path))
            if date not in lst:
                os.makedirs('{}/StockIntraDayData/{}'.format(self.data_path, date))
                for stock in all_stocks:
                    intradaydata = get_price(stock,frequency='minute',date=date)
                    with open('{}/StockIntraDayData/{}/{}.pkl'.format(self.data_path, date, stock), 'wb') as f:
                        pickle.dump(intradaydata, f)
