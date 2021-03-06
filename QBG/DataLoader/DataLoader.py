# Copyright (c) 2021 Dai HBG

"""
该代码定义的类用于调用聚宽的接口查询数据
This code is used for calling the interface of Join Quant to query data

v1.0
2021-08-30
-- 暂时不支持分钟数据，因为聚宽的数据查询有次数限制
-- get_pv_data方法的参数data_type传入一个列表，表示需要的数据
-- 原始的数据读出来后构造成一个字典，关键字就是原先的列名，值是按照日期排列的矩阵

2021-09-01
-- 更新：每日量价数据merge成一个dic

2021-09-04
-- 更新：get_matrix_data方法返回的Data类还要包括factor和位置到股票位置的映射
-- 更新：Data类新增根据输入起始日期找到有交易的最近的两个真正的起始日期并返回的方法

2021-09-05
-- 更新：Data类新增position_date_dic属性，用于检测每一个交易日给出的股票仓位

2021-09-07
-- 更新：在调用get_matrix_data方法时，检查文件夹下是否包含了所需的所有文件，只有都包含，且检查开始日期和结束日期包含在之中时才读出，否则
        全部重新生成

2021-09-12
-- 更新：Data类生成时加入一个raw_ret属性，用于记录原始的日收益率，判断是否是涨停板不可买入
-- 更新：top的计算加入流动性选择、股票池选择，例如中证500
-- 更新：get_pv_data新增获取每日的行业和概念数据的字段，可以获取每天的所有概念和所有行业分类的股票，目前对于行业只使用二级行业分类，以节省查询行数

2021-09-22
-- 更新：get_pv_data获取分钟数据时，采用增量更新，已有的数据不再重复查询，节约查询行数

2021-09-28
-- 不同频率的日内数据分开存放，为了提高数据获取量，现在默认获取10min数据

2021-10-01
-- DataFrame类型的数据默认写入方式为csv，避免pandas的版本不同而无法读入
"""

import numpy as np
import pandas as pd
import os
import pickle
import jqdatasdk
from jqdatasdk import *
import datetime


class Data:
    def __init__(self, code_order_dic, order_code_dic, date_position_dic, position_date_dic,
                 data_dic, ret, industry, start_date, end_date, top):
        """
        :param code_order_dic: 股票代码到矩阵位置的字典
        :param order_code_dic: 矩阵位置到股票代码的字典
        :param date_position_dic: 日期到矩阵下标的字典
        :param data_dic: 所有的数据，形状一致
        :param ret: 使用的收益率
        :param industry: 使用的行业分类，是一个字典，值是一个矩阵，里面的数字就是行业分类
        :param start_date: 回测开始日期
        :param end_date: 回测结束日期
        :param top: top矩阵中存储每一个交易日可选的股票
        """
        self.code_order_dic = code_order_dic
        self.order_code_dic = order_code_dic
        self.date_position_dic = date_position_dic
        self.position_date_dic = position_date_dic
        self.data_dic = data_dic
        self.ret = ret
        self.industry = industry
        self.start_date = start_date
        self.end_date = end_date
        self.top = top

    def get_real_date(self, start_date, end_date):
        """
        :param start_date: 任意输入的开始日期
        :param end_date: 任意输入的结束日期
        :return: 返回有交易的真正的起始日期对应的下标
        """
        tmp_start = start_date.split('-')
        i = 0
        while True:
            s = datetime.date(int(tmp_start[0]), int(tmp_start[1]), int(tmp_start[2])) + datetime.timedelta(days=i)
            try:
                start = self.date_position_dic[s]
                break
            except KeyError:
                i += 1
        i = 0
        tmp_end = end_date.split('-')
        while True:
            s = datetime.date(int(tmp_end[0]), int(tmp_end[1]), int(tmp_end[2])) + datetime.timedelta(days=i)
            try:
                end = self.date_position_dic[s]
                break
            except KeyError:
                i -= 1
        return start, end


class DataLoader:
    def __init__(self, user_id, password, data_path='F:/Documents/AutoFactoryData',
                 back_test_data_path='F:/Documents/AutoFactoryData/BackTestData'):
        """
        :param user_id: 登录聚宽的用户id
        :param password: 登录密码
        :param data_path: 存放数据的路径
        :param back_test_data_path: 回测数据的存放路径
        """
        self.data_path = data_path
        self.back_test_data_path = back_test_data_path  # 该路径用于存放某一次回测所需要的任何字典
        self.user_id = user_id
        self.password = password
        jqdatasdk.auth(self.user_id, self.password)  # 登录聚宽

    """
    get_pv_data定义了从聚宽读取量价数据(pv for Price & Volume)并保存在本地的方法
    """

    def get_pv_data(self, start_date, end_date, data_type=None):  # 获得日频量价关系数据
        """
        :param start_date: 开始日期
        :param end_date: 结束日期，增量更新时这两个值设为相同
        :param data_type: 数据类型，stock_daily表示日频股票
        :return: 无返回值
        """
        if data_type is None:  # 参数默认值不要是可变的，否则可能出错
            data_type = ['stock_daily', 'industry']  # 默认获取股票日数据，行业和概念分类

        start_date = start_date.split('-')
        end_date = end_date.split('-')

        begin = datetime.date(int(start_date[0]), int(start_date[1]), int(start_date[2]))
        end = datetime.date(int(end_date[0]), int(end_date[1]), int(end_date[2]))

        all_stocks = list(get_all_securities(types=['stock'], date=end_date).index)  # 只获取最后一天的

        if 'stock_daily' in data_type:  # 获取日频量价、资金流数据

            lst = os.listdir('{}/StockDailyData'.format(self.data_path))  # 所有的日期文件夹

            for i in range((end - begin).days + 1):
                date = begin + datetime.timedelta(days=i)
                if date.weekday() in [5, 6]:  # 略过周末
                    continue

                # 获得价格数据
                stock_data = get_price(all_stocks, frequency='daily',
                                       fields=['open', 'close', 'low', 'high', 'volume', 'money', 'pre_close',
                                               'factor', 'avg'],
                                       start_date=date, end_date=date)
                if len(stock_data) == 0:  # 判断当天有无交易
                    continue
                if str(date) not in lst:
                    os.makedirs('{}/StockDailyData/{}'.format(self.data_path, date))
                stock_data.index = stock_data['code']
                # 获得资金流数据
                money_flow = get_money_flow(all_stocks, start_date=date, end_date=date)
                money_flow.index = money_flow['sec_code']
                # 获取财务数据
                fundamental = get_fundamentals(query(valuation, indicator), date=date)
                fundamental.index = fundamental['code']
                money_flow.to_csv('{}/StockDailyData/{}/money_flow_{}.csv'.format(self.data_path, date, date),
                                  index=False)
                fundamental.to_csv('{}/StockDailyData/{}/fundamental_{}.csv'.format(self.data_path, date, date),
                                  index=False)
                stock_data.to_csv('{}/StockDailyData/{}/stock_{}.csv'.format(self.data_path, date, date),
                                  index=False)
                print('{} done.'.format(date))

        if 'index_daily' in data_type:
            begin = datetime.date(int(start_date[0]), int(start_date[1]), int(start_date[2]))
            end = datetime.date(int(end_date[0]), int(end_date[1]), int(end_date[2]))
            for i in range((end - begin).days + 1):
                date = begin + datetime.timedelta(days=i)
                if date.weekday() in [5, 6]:  # 略过周末
                    continue
                all_indexes = get_all_securities(types=['index'], date=date)
                all_indexes.to_csv('{}/StockDailyData/{}/index_{}.csv'.format(self.data_path, date, date),
                                  index=False)
                print('{} done.'.format(date))

        if 'industry' in data_type:  # 获取行业分类，最后以字典形式存储
            print('getting industry data...')
            concepts = list(get_concepts().index)  # 获得所有的概念名称
            begin = datetime.date(int(start_date[0]), int(start_date[1]), int(start_date[2]))
            end = datetime.date(int(end_date[0]), int(end_date[1]), int(end_date[2]))
            for i in range((end - begin).days + 1):
                date = begin + datetime.timedelta(days=i)
                if date.weekday() in [5, 6]:  # 略过周末
                    continue
                industry_dic = {'concept': {}, 'swf': {}, 'sws': {}, 'swt': {}}
                ind = list(get_industries('sw_l1').index)  # 申万一级行业
                for name in ind:
                    industry_dic['swf'][name] = get_industry_stocks(name, date=date)
                ind = list(get_industries('sw_l2').index)  # 申万二级行业
                for name in ind:
                    industry_dic['sws'][name] = get_industry_stocks(name, date=date)
                ind = list(get_industries('sw_l3').index)  # 申万三级行业
                for name in ind:
                    industry_dic['swt'][name] = get_industry_stocks(name, date=date)
                for name in concepts:
                    industry_dic['concept'][name] = get_concept_stocks(name, date=date)
                with open('{}/StockDailyData/{}/industry_{}.pkl'.format(self.data_path, date, date), 'wb') as f:
                    pickle.dump(industry_dic, f)
                print('{} done.'.format(date))

        if '1m' in data_type:
            lst = os.listdir('{}/StockIntraDayData/1m'.format(self.data_path))
            for i in range((end - begin).days + 1):
                date = begin + datetime.timedelta(days=i)
                if date.weekday() in [5, 6]:  # 略过周末
                    continue
                if str(date) not in lst:
                    os.makedirs('{}/StockIntraDayData/1m/{}'.format(self.data_path, date))
                stocks = os.listdir('{}/StockIntraDayData/1m/{}'.format(self.data_path, date))  # 上一次查询已有的股票
                for stock in all_stocks:  # 剔除创业板股票，避免超出查询限制
                    if stock[:3] == '300' or stock[:3] == '688':
                        continue
                    if '{}.pkl'.format(stock) in stocks:
                        continue
                    end_date = date + datetime.timedelta(days=1)
                    intra_day_data = get_price(stock, frequency='1m',
                                               fields=['open', 'close', 'low', 'high', 'volume', 'money', 'pre_close',
                                                       'factor'],
                                               start_date=date, end_date=end_date)
                    intra_day_data.to_csv('{}/StockIntraDayData/1m/{}/{}.csv'.format(self.data_path, date, stock),
                                          index=False)

        if '10m' in data_type:
            lst = os.listdir('{}/StockIntraDayData/10m'.format(self.data_path))
            for i in range((end - begin).days + 1):
                date = begin + datetime.timedelta(days=i)
                if date.weekday() in [5, 6]:  # 略过周末
                    continue
                if str(date) not in lst:
                    os.makedirs('{}/StockIntraDayData/10m/{}'.format(self.data_path, date))
                stocks = os.listdir('{}/StockIntraDayData/10m/{}'.format(self.data_path, date))  # 上一次查询已有的股票
                for stock in all_stocks:  # 剔除创业板股票，避免超出查询限制
                    if stock[:3] == '300' or stock[:3] == '688':
                        continue
                    if '{}.csv'.format(stock) in stocks:
                        continue
                    end_date = date + datetime.timedelta(days=1)
                    intra_day_data = get_price(stock, frequency='10m',
                                               fields=['open', 'close', 'low', 'high', 'volume', 'money'],
                                               start_date=date, end_date=end_date)
                    intra_day_data.to_csv('{}/StockIntraDayData/10m/{}/{}.csv'.format(self.data_path, date, stock),
                                          index=False)

    """
    get_matrix_data方法读取给定起始日期的原始数据，并生成需要的收益率矩阵，字典等
    
    v1.0
    2021-08-30
    -- 需要解析一个字段，告诉DataLoader应该取从什么地方到什么地方的收益率，例如开盘价收益率或者日内收益率，周收益率等
        a. 具有形式"{}_{}_{}".format(data, data, day)的形式，表示从其中一个数据到另一个数据中间间隔day天
           例如open_close_4表示周一开盘价到周五收盘价
    2021-08-31
    1. 向前回溯例天数，默认100天，然后还要获得一个日期和下标对应的字典，以确定回测时的对应
    2021-09-02
    -- get_matrix_data方法应当返回一个data，是一个Data类，包含了所需的调用信息，以后在各类之间交流信息更方便
    2021-09-04
    -- get_matrix_data方法需要剔除科创版和创业板股票，并且返回Data类中需要写入top矩阵
    """

    def get_matrix_data(self, back_test_name='default', frequency=None,
                        start_date='2021-01-01', end_date='2021-06-30', back_windows=10,
                        return_type='close_close_1', top_constraint='volume', need_industry=False):
        # 在获取足够多的行业数据之前要通过字段确定是否要加入industry字段
        """
        :param need_industry: 是否需要处理行业信息，在获得足够多行业信息后将删除
        :param back_test_name: 该回测的名字
        :param frequency: 回测频率，目前默认且仅支持日频
        :param start_date: 回测开始时间
        :param end_date: 结束时间
        :param back_windows: 开始时间向前多长的滑动窗口
        :param return_type: 该字段描述需要预测的收益率类型
        :param top_constraint: 决定如何筛选股票，默认选出成交量最大的50只股票
        :return:
        """
        # 读入dataframe数据另存为便于处理的矩阵形式
        tmp = start_date.split('-')
        start_date = datetime.date(int(tmp[0]), int(tmp[1]), int(tmp[2]))  # 回测开始时间
        tmp = end_date.split('-')
        end_date = datetime.date(int(tmp[0]), int(tmp[1]), int(tmp[2]))  # 回测结束时间

        if frequency is None:
            frequency = ['daily']
        data = None  # 如果最后没有任何操作，就返回None
        if 'daily' in frequency:  # 当前仅支持日频的回测
            lst = os.listdir('{}'.format(self.back_test_data_path))
            if back_test_name not in lst:
                os.makedirs('{}/{}'.format(self.back_test_data_path, back_test_name))
            lst = os.listdir('{}/{}'.format(self.back_test_data_path, back_test_name))
            names_to_check = ['code_order_dic.pkl', 'raw_data_dic.pkl', 'return.pkl',
                              'order_code_dic.pkl', 'date_position_dic.pkl',
                              'position_date_dic.pkl', 'top.pkl', 'start_end_date.pkl']
            if need_industry:
                names_to_check += ['industry.pkl', 'industry_order_dic.pkl', 'order_industry_dic.pkl']
            # 判断是否要重写
            rewrite = False
            for name in names_to_check:
                if name not in lst:
                    print('{} not found'.format(name))
                    rewrite = True
                    break
            if not rewrite:
                with open('{}/{}/start_end_date.pkl'.format(self.back_test_data_path, back_test_name), 'rb') as f:
                    start_end_date = pickle.load(f)
                if start_date < start_end_date[0] or end_date > start_end_date[1]:
                    rewrite = True

            if rewrite:  # code_order_dic用于存储该回测区间内出现过的股票代码到矩阵位置的映射
                print('getting data...')
                start_end_date = (start_date, end_date)
                with open('{}/{}/start_end_date.pkl'.format(self.back_test_data_path, back_test_name), 'wb') as f:
                    pickle.dump(start_end_date, f)
                dates = os.listdir('{}/StockDailyData'.format(self.data_path))
                code_order_dic = {}  # 股票代码到位置的映射
                order_code_dic = {}  # 位置到股票代码的映射
                order = 0
                days = 0
                date_position_dic = {}  # 记录日期对应到数据矩阵的位置
                position_date_dic = {}  # 记录对应数据矩阵位置到日期的映射
                length = int(return_type.split('_')[-1])  # 表示需要延后几天以获得对应的收益
                for i in range(-back_windows, (end_date - start_date).days + 1 + length + 1 + 2):
                    date = start_date + datetime.timedelta(days=i)  # 这里有bug要修复，万一延后的两天是周末，就有问题。加两天保险
                    if date.weekday() in [5, 6]:  # 周末略过
                        continue
                    if str(date) in dates:
                        date_position_dic[date] = days  # 这个日期对应的矩阵第几行
                        position_date_dic[days] = date  # 第几行对应的是哪一天的日期
                        data = pd.read_csv('{}/StockDailyData/{}/stock_{}.pkl'.format(self.data_path,
                                                                             date, date))

                            if len(data) == 0:  # 说明当前无交易，略过
                                continue
                            codes = list(data['code'])
                            for code in codes:
                                if code[:3] in ['688', '300']:  # 剔除创业板和科创版的股票
                                    continue
                                try:
                                    code_order_dic[code]
                                except KeyError:  # 代码规范：最好写明具体的错误类型
                                    code_order_dic[code] = order
                                    order_code_dic[order] = code
                                    order += 1
                        days += 1
                with open('{}/{}/code_order_dic.pkl'.format(self.back_test_data_path, back_test_name), 'wb') as f:
                    pickle.dump(code_order_dic, f)
                with open('{}/{}/order_code_dic.pkl'.format(self.back_test_data_path, back_test_name), 'wb') as f:
                    pickle.dump(order_code_dic, f)
                with open('{}/{}/date_position_dic.pkl'.format(self.back_test_data_path, back_test_name), 'wb') as f:
                    pickle.dump(date_position_dic, f)
                with open('{}/{}/position_date_dic.pkl'.format(self.back_test_data_path, back_test_name), 'wb') as f:
                    pickle.dump(position_date_dic, f)

                # 获得行业字典，形状和数据字典一致，需要构造一个行业到数字的映射
                industry_order_dic = None
                if need_industry:
                    industry_order_dic = {'swf': {}, 'sws': {}, 'swt': {}, 'concept': {}}  # 行业编号到对应序号的字典，每个独立
                    order_industry_dic = {'swf': {}, 'sws': {}, 'swt': {}, 'concept': {}}  # 对应序号到行业编号的字典，每个独立
                    num_dic = {'swf': 0, 'sws': 0, 'swt': 0, 'concept': 0}  # 记录已经出现过的行业编号的总数
                    for i in range(-back_windows, (end_date - start_date).days + 1 + length + 1 + 2):
                        date = start_date + datetime.timedelta(days=i)  # 这里有bug要修复，万一延后的两天是周末，就有问题。加两天保险
                        if date.weekday() in [5, 6]:  # 周末略过
                            continue
                        if str(date) in dates:
                            with open('{}/StockDailyData/{}/industry_{}.pkl'.format(self.data_path,
                                                                                    date, date), 'rb') as file:
                                industry_dic = pickle.load(file)
                                for key, value in industry_dic.items():  # value也是一个字典，key是行业编号，value是股票列表
                                    ind_names = list(value.keys())  # 行业编号
                                    for name in ind_names:
                                        try:
                                            industry_order_dic[key][name]
                                        except KeyError:
                                            industry_order_dic[key][name] = num_dic[key]  # 该行业分类准则下一个行业编号的序号
                                            order_industry_dic[key][num_dic[key]] = name
                                            num_dic[key] += 1
                    with open('{}/{}/industry_order_dic.pkl'.format(self.back_test_data_path, back_test_name),
                              'wb') as f:
                        pickle.dump(industry_order_dic, f)
                    with open('{}/{}/order_industry_dic.pkl'.format(self.back_test_data_path, back_test_name),
                              'wb') as f:
                        pickle.dump(order_industry_dic, f)

                # 获得数据字典
                names = ['open', 'close', 'high', 'low', 'avg', 'factor', 'volume', 'turnover_ratio',
                         'net_pct_main', 'net_pct_xl', 'net_pct_l', 'net_pct_m', 'net_pct_s']

                data_dic = {}  # 原始数据字典
                for name in names:
                    data_dic[name] = np.zeros((days, len(code_order_dic)))

                industry = {}  # 行业分类字典
                ind_names = ['swf', 'sws', 'swt', 'concept']  # 行业分类准则
                for name in ind_names:
                    industry[name] = -np.ones((days, len(code_order_dic)))  # 初始化为-1，如果有股票不被分类

                ret = np.zeros((days, len(code_order_dic)))
                start_name = return_type.split('_')[0]
                end_name = return_type.split('_')[1]

                k = 0
                for i in range(-back_windows, (end_date - start_date).days + 1 + length + 1):
                    date = start_date + datetime.timedelta(days=i)
                    if date.weekday() in [5, 6]:
                        continue
                    if str(date) in dates:
                        # 处理基本数据
                        data = pd.read_csv('{}/StockDailyData/{}/stock_{}.pkl'.format(self.data_path,
                                                                             date, date))
                        if len(data) == 0:
                            continue
                        index = list(data['code'])
                        for j in range(len(data)):
                            if index[j][:3] in ['688', '300']:  # 剔除科创版和创业板股票
                                continue
                            for name in names[:7]:
                                data_dic[name][k, code_order_dic[index[j]]] = data[name].iloc[j]

                        # 处理基本面
                        data = pd.read_csv('{}/StockDailyData/{}/fundamental_{}.pkl'.format(self.data_path,
                                                                                   date, date))
                        index = list(data.index)
                        for j in range(len(data)):
                            if index[j][:3] in ['688', '300']:  # 剔除科创版和创业板股票
                                continue
                            for name in names[7:8]:
                                try:
                                    data_dic[name][k, code_order_dic[index[j]]] = data[name].iloc[j]
                                except KeyError:
                                    pass

                        # 处理资金流
                        data = pd.read_csv('{}/StockDailyData/{}/money_flow_{}.pkl'.format(self.data_path,
                                                                                  date, date))
                        index = list(data.index)
                        for j in range(len(data)):
                            if index[j][:3] in ['688', '300']:  # 剔除科创版和创业板股票
                                continue
                            for name in names[8:]:
                                data_dic[name][k, code_order_dic[index[j]]] = data[name].iloc[j]

                        # 处理行业
                        if need_industry:
                            with open('{}/StockDailyData/{}/industry_{}.pkl'.format(self.data_path,
                                                                                    date, date), 'rb') as file:
                                data = pickle.load(file)
                                for ind_name in ind_names:
                                    ind = data[ind_name]  # 该天的一个行业分类，形式是行业编号：代码列表
                                    for key, value in ind.items():  # value是一个列表，里面是股票代码
                                        ind_num = industry_order_dic[ind_name][key]

                                        for code in value:  # 这个列表里面的股票代码
                                            try:
                                                industry[ind_name][k, code_order_dic[code]] = ind_num
                                            except KeyError:
                                                pass

                        print('{} done.'.format(date))
                        k += 1
                ret[:-length] = data_dic[end_name][length:] / data_dic[start_name][:-length] - 1
                ret[np.isnan(ret)] = 0

                # 生成top
                top = (data_dic['close'] < 100) & (data_dic['close'] > 10)
                for i in range(len(top) - 1, 0, -1):  # 剔除上市不足50个交易日的股票
                    for j in range(top.shape[1]):
                        if i <= 50:
                            if np.isnan(data_dic['close'][0, j]) or data_dic['close'][0, j] == 0:
                                top[i, j] = False
                        else:
                            if np.isnan(data_dic['close'][i - 50, j]) or data_dic['close'][i - 50, j] == 0:
                                top[i, j] = False

                if top_constraint == 'volume':  # 按照成交量筛选前1000的股票
                    for i in range(len(top)):
                        tmp = data_dic['volume'][i][top[i]].argsort()[-1000:]  # 成交量最大的1000只
                        tmp_value = np.zeros(np.sum(top[i]))
                        tmp_value[tmp] = True
                        top[i][top[i]] = tmp_value

                # 写入数据
                with open('{}/{}/raw_data_dic.pkl'.format(self.back_test_data_path, back_test_name), 'wb') as f:
                    pickle.dump(data_dic, f)
                with open('{}/{}/return.pkl'.format(self.back_test_data_path, back_test_name), 'wb') as f:
                    pickle.dump(ret, f)
                with open('{}/{}/top.pkl'.format(self.back_test_data_path, back_test_name), 'wb') as f:
                    pickle.dump(top, f)
                if need_industry:
                    with open('{}/{}/industry.pkl'.format(self.back_test_data_path, back_test_name), 'wb') as f:
                        pickle.dump(industry, f)
                    data = Data(code_order_dic, order_code_dic, date_position_dic, position_date_dic,
                                data_dic, ret, industry, start_date, end_date, top)
                else:
                    data = Data(code_order_dic, order_code_dic, date_position_dic, position_date_dic,
                                data_dic, ret, industry=None, start_date=start_date, end_date=end_date, top=top)
            else:
                # 直接读入数据
                print('using cache')
                with open('{}/{}/raw_data_dic.pkl'.format(self.back_test_data_path, back_test_name), 'rb') as f:
                    data_dic = pickle.load(f)
                with open('{}/{}/return.pkl'.format(self.back_test_data_path, back_test_name), 'rb') as f:
                    ret = pickle.load(f)
                with open('{}/{}/code_order_dic.pkl'.format(self.back_test_data_path, back_test_name), 'rb') as f:
                    code_order_dic = pickle.load(f)
                with open('{}/{}/order_code_dic.pkl'.format(self.back_test_data_path, back_test_name), 'rb') as f:
                    order_code_dic = pickle.load(f)
                with open('{}/{}/date_position_dic.pkl'.format(self.back_test_data_path, back_test_name), 'rb') as f:
                    date_position_dic = pickle.load(f)
                with open('{}/{}/position_date_dic.pkl'.format(self.back_test_data_path, back_test_name), 'rb') as f:
                    position_date_dic = pickle.load(f)
                with open('{}/{}/top.pkl'.format(self.back_test_data_path, back_test_name), 'rb') as f:
                    top = pickle.load(f)
                if need_industry:
                    with open('{}/{}/industry.pkl'.format(self.back_test_data_path, back_test_name), 'rb') as f:
                        industry = pickle.load(f)
                    data = Data(code_order_dic, order_code_dic, date_position_dic, position_date_dic,
                                data_dic, ret, industry, start_date, end_date, top)
                else:
                    data = Data(code_order_dic, order_code_dic, date_position_dic, position_date_dic,
                                data_dic, ret, industry=None, start_date=start_date, end_date=end_date, top=top)

            if '10m' in frequency:  # 读取10min数据
                if 'daily' not in frequency or data is None:
                    print('daily data is needed!')
                    return
                lst = os.listdir('{}'.format(self.back_test_data_path))
                if back_test_name not in lst:
                    os.makedirs('{}/{}'.format(self.back_test_data_path, back_test_name))
                lst = os.listdir('{}/{}'.format(self.back_test_data_path, back_test_name))
                names_to_check = ['intra_open.pkl', 'intra_high.pkl', 'intra_low.pkl', 'intra_close.pkl',
                                  'intra_volume.pkl', 'intra_money.pkl', 'intra_avg.pkl']
                # 判断是否要重写
                for name in names_to_check:
                    if name not in lst:
                        print('{} not found'.format(name))
                        rewrite = True
                        break
                names = ['intra_open', 'intra_high', 'intra_low', 'intra_close', 'intra_volume',
                         'intra_money', 'intra_avg']
                # 判断是否要重写
                if rewrite:  # 需要重写数据
                    data.data_dic['intra_close'] = np.zeros((data.data_dic['close'].shape[0], 24,
                                                             data.data_dic['close'].shape[1]))
                    data.data_dic['intra_open'] = np.zeros((data.data_dic['close'].shape[0], 24,
                                                            data.data_dic['close'].shape[1]))
                    data.data_dic['intra_high'] = np.zeros((data.data_dic['close'].shape[0], 24,
                                                            data.data_dic['close'].shape[1]))
                    data.data_dic['intra_low'] = np.zeros((data.data_dic['close'].shape[0], 24,
                                                           data.data_dic['close'].shape[1]))
                    data.data_dic['intra_volume'] = np.zeros((data.data_dic['close'].shape[0], 24,
                                                              data.data_dic['close'].shape[1]))
                    data.data_dic['intra_money'] = np.zeros((data.data_dic['close'].shape[0], 24,
                                                             data.data_dic['close'].shape[1]))
                    data.data_dic['intra_avg'] = np.zeros((data.data_dic['close'].shape[0], 24,
                                                           data.data_dic['close'].shape[1]))
                    dates = os.listdir('{}/StockDailyData'.format(self.data_path))
                    length = int(return_type.split('_')[-1])
                    k = 0
                    print('getting 10m data...')
                    for i in range(-back_windows, (end_date - start_date).days + 1 + length + 1):
                        date = start_date + datetime.timedelta(days=i)
                        if date.weekday() in [5, 6]:
                            continue
                        if str(date) in dates:
                            stocks = os.listdir('{}/StockIntraDayData/10m/{}'.format(self.data_path, date))
                            for stock in stocks:  # 依次读入每一只股票
                                with open('{}/StockIntraDayData/10m/{}/{}'.format(self.data_path,
                                                                                  date, stock), 'rb') as file:
                                    intra_data = pickle.load(file)
                                    try:
                                        for name in names:
                                            data.data_dic[name][k, :, code_order_dic[stock[:-4]]] = \
                                                intra_data[name[6:]].values
                                        data.data_dic['intra_avg'][k, :, code_order_dic[stock[:-4]]] = \
                                            intra_data['money'].values / intra_data['volume'].values
                                    except KeyError:
                                        pass
                            k += 1
                            print('{} done.'.format(date))
                    for name in names:
                        with open('{}/{}/{}.pkl'.format(self.back_test_data_path, back_test_name, name), 'wb') as f:
                            pickle.dump(data.data_dic[name], f)
                else:
                    print('using cache')
                    for name in names:
                        with open('{}/{}/{}.pkl'.format(self.back_test_data_path, back_test_name, name), 'rb') as f:
                            data.data_dic[name] = pickle.load(f)
        return data
