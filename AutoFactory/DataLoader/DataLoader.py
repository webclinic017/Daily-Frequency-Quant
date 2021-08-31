# Copyright (c) 2021 Dai HBG

"""
该代码定义的类用于调用聚宽的接口查询数据
This code is used for calling the interface of Join Quant to query data

v1.0
2021-08-30
-- 暂时不支持分钟数据，因为聚宽的数据查询有次数限制
-- get_pv_data方法的参数data_type传入一个列表，表示需要的数据
-- 原始的数据读出来后构造成一个字典，关键字就是原先的列名，值是按照日期排列的矩阵
"""

import numpy as np
import os
import pandas as pd
import pickle
import jqdatasdk
from jqdatasdk import *
import datetime


class DataLoader:
    def __init__(self, user_id, password, data_path='F:/Documents/AutoFactoryData',
                 back_test_data_path='F:/Documents/AutoFactoryData/BackTestData'):
        self.data_path = data_path
        self.back_test_data_path = back_test_data_path  # 该路径用于存放某一次回测所需要的任何字典
        self.user_id = user_id
        self.password = password
        jqdatasdk.auth(self.user_id, self.password)  # 登录聚宽

    """
    get_pv_data定义了从聚宽读取量价数据(pv for Price & Volume)并保存在本地的方法
    """

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

    """
    get_matrix_data方法读取给定起始日期的原始数据，并生成需要的收益率矩阵，字典等
    
    v1.0
    2021-08-30
    1. 需要解析一个字段，告诉DataLoader应该取从什么地方到什么地方的收益率，例如开盘价收益率或者日内收益率，周收益率等
    """

    def get_matrix_data(self, back_test_name='default', frequency='daily',
                        start_date='2021-01-01', end_date='2021-06-30'):  # 读入dataframe数据另存为便于处理的矩阵形式
        tmp = start_date.split('-')
        start_date = datetime.date(int(tmp[0]), int(tmp[1]), int(tmp[2]))  # 回测开始时间
        tmp = end_date.split('-')
        end_date = datetime.date(int(tmp[0]), int(tmp[1]), int(tmp[2]))  # 回测结束时间
        if frequency == 'daily':  # 当前仅支持日频的回测
            lst = os.listdir('{}'.format(self.back_test_data_path))
            if back_test_name not in lst:
                os.makedirs('{}/{}'.format(self.back_test_data_path, back_test_name))
            lst = os.listdir('{}/{}'.format(self.back_test_data_path, back_test_name))
            if 'code_order_dic.pkl' not in lst:  # code_order_dic用于存储该回测区间内出现过的股票代码到矩阵位置的映射
                dates = os.listdir('{}/StockDailyData'.format(self.data_path))
                codes_order_dic = {}
                order = 0
                days = 1
                for i in range((end_date - start_date).days + 1):
                    date = start_date + datetime.timedelta(days=i)
                    if str(date) in dates:  # 要加入判断今天是否全市场停牌，方法是检测相邻两天是不是完全一致
                        with open('{}/StockDailyData/{}/stock_{}.pkl'.format(self.data_path,
                                                                             date, date), 'rb') as file:
                            data = pickle.load(file)
                            codes = list(data.index)
                            for code in codes:
                                try:
                                    codes_order_dic[code]
                                except KeyError as e:  # 代码规范：最好写明具体的错误类型
                                    codes_order_dic[code] = order
                                    order += 1
                        days += 1
                with open('{}/{}/code_order_dic.pkl'.format(self.back_test_data_path, back_test_name), 'wb') as f:
                    pickle.dump(codes_order_dic, f)
            else:
                with open('{}/{}/code_order_dic.pkl'.format(self.back_test_data_path, back_test_name), 'rb') as f:
                    codes_order_dic = pickle.load(f)
                dates = os.listdir('{}/StockDailyData'.format(self.data_path))
                days = 1
                for i in range((end_date - start_date).days + 1):
                    date = start_date + datetime.timedelta(days=i)
                    if str(date) in dates:
                        days += 1

            names = ['open', 'close', 'high', 'low', 'volume', 'money']
            data_dic = {}
            for name in names:
                data_dic[name] = np.zeros((days, len(codes_order_dic)))

            ret = np.zeros((days, len(codes_order_dic)))

            k = 0
            for i in range((end_date - start_date).days + 1):
                date = start_date + datetime.timedelta(days=i)
                if str(date) in dates:
                    with open('{}/StockDailyData/{}/stock_{}.pkl'.format(self.data_path,
                                                                         date, date), 'rb') as file:
                        data = pickle.load(file)
                        index = list(data.index)
                        for j in range(len(data)):
                            for name in names:
                                data_dic[name][k, codes_order_dic[index[j]]] = data[name][j]
                            ret[k, codes_order_dic[index[j]]] = data['close'][j] / data['open'][j] - 1  # 需要改成定制的收益率
                        k += 1
            with open('{}/{}/raw_data_dic.pkl'.format(self.back_test_data_path, back_test_name), 'wb') as f:
                pickle.dump(data_dic, f)
            with open('{}/{}/return.pkl'.format(self.back_test_data_path, back_test_name), 'wb') as f:
                pickle.dump(ret, f)