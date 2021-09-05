# Copyright (c) 2021 Dai HBG

"""
AutoFactory类是一个总体的集成类，通过调用其他类实现以下功能模块：
1. 测试因子，该功能通过AutoFormula类定义的方法实现
2. 自动搜寻因子，该功能通过调用AutoFormula类定义的方法实现
3. 模型训练，该功能通过调用Model类定义的各种结构化模型实现
4. 模拟交易，该功能通过调用Trader类定义的方法进行信号交易，评估效果
5. 股票生成，该功能通过调用real模式的Trader，根据日期给出股票信号，从高到低排列

结构说明：
1. AutoFactory类是核心，其初始化的时候将调用DataLoader读取数据，然后得到data_dic等保存在类属性中
2. AutoFormula文件夹下定义的都是和formula相关的类，因此所有和公式有关的方法，包括树的生成，解析，信号提取等，都应该在里面定义
3. Tester文件夹下定义的类都是和信号评价相关的类，因此所有和信号相关的方法都应该在里面定义，包括测试信号表现的
4. Model文件夹下定义的类都是和模型定义相关的
"""
import numpy as np
import sys
import os
import pickle

sys.path.append('C:/Users/Administrator/Desktop/Daily-Frequency-Quant/AutoFactory/Tester/')
sys.path.append('C:/Users/Administrator/Desktop/Daily-Frequency-Quant/AutoFactory/DataLoader/')
sys.path.append('C:/Users/Administrator/Desktop/Daily-Frequency-Quant/AutoFactory/AutoFormula/')
sys.path.append('C:/Users/Administrator/Desktop/Daily-Frequency-Quant/AutoFactory/Model/')

from DataLoader import DataLoader
from BackTester import BackTester
from AutoFormula import AutoFormula
import Model


class AutoFactory:
    def __init__(self, user_id, password, start_date, end_date, dump_signal_path=None, return_type='open_close_4'):
        """
        :param user_id: 登录聚宽的用户id
        :param password: 登录密码
        :param start_date: 总体的开始日期
        :param end_date: 总体的结束日期
        :param dump_signal_path: dump信号矩阵路径
        """
        self.start_date = start_date
        self.end_date = end_date
        self.dataloader = DataLoader(user_id, password)
        self.data = self.dataloader.get_matrix_data(start_date=start_date,
                                                    end_date=end_date, return_type=return_type)
        self.back_tester = BackTester(data=self.data)  # 模拟交易回测
        self.autoformula = AutoFormula(start_date=start_date, end_date=end_date, top=self.data.top)

        if dump_signal_path is None:
            lst = os.listdir('F:/Documents/AutoFactoryData/Signal')
            if '{}-{}'.format(start_date, end_date) not in lst:
                os.makedirs('F:/Documents/AutoFactoryData/Signal/{}-{}'.format(start_date, end_date))
            dump_signal_path = 'F:/Documents/AutoFactoryData/Signal/{}-{}'.format(start_date, end_date)
        self.dump_signal_path = dump_signal_path

        self.dump_factor_path = 'F:/Documents/AutoFactoryData/Factors'

    def test_factor(self, formula, start_date=None, end_date=None, prediction_mode=False):  # 测试因子
        """
        :param formula: 回测的公式
        :param start_date: 回测开始日期
        :param end_date: 回测结束日期
        :param prediction_mode: 是否是最新预测模式，是的话不需要测试，只生成signal
        :return: 返回统计类的实例
        """
        if not prediction_mode:
            if start_date is None:
                start_date = self.start_date
            if end_date is None:
                end_date = self.end_date
            stats, signal = self.autoformula.test_formula(formula, self.data, start_date, end_date)
            # return stats,s,e
            print('mean IC: {:.4f}, auto_corr: {:.4f}, positive_IC_rate: {:.4f}, IC_IR: {:.4f}'. \
                  format(stats.mean_IC, stats.auto_corr, stats.positive_IC_rate, stats.IC_IR))
            return stats, signal
        else:
            return self.autoformula.test_formula(formula, self.data, start_date, end_date,
                                                 prediction_mode=prediction_mode)  # 只返回signal

    def backtest(self, signal, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

    def dump_signal(self, signal):
        num = len(os.listdir(self.dump_signal_path))  # 先统计有多少信号
        with open('{}/signal_{}.pkl'.format(self.dump_signal_path, num), 'wb') as f:
            pickle.dump(signal, f)

    def dump_factor(self, factor, path=None):
        if path is None:
            path = '{}/factors_pv_{}_{}.txt'.format(self.dump_factor_path, self.data.start_date, self.data.end_date)
        else:
            path = '{}/{}.txt'.format(self.dump_factor_path, path)
        with open(path, 'a+') as f:
            f.write(factor + '\n')

    def long_stock_predict(self, model_name, n=1):  # 每日推荐股票多头
        """
        :param date: 用到多少天前的数据，默认以下一个交易日收盘价买入
        :param n: 推荐得分最高的n只股票
        :param model_name: 使用的模型
        :return: 直接打印结果
        """
        date = str(self.data.end_date)  # 这里之后的版本要修改成更加灵活的读写信号，例如每天自动增量更新
        print('reading model...')
        with open('F:/Documents/AutoFactoryData/Model/{}.pkl'.format(model_name), 'rb') as file:
            model = pickle.load(file)
        with open('F:/Documents/AutoFactoryData/Factors/factors_pv_2020-11-01_2021-07-30.txt') as file:
            while True:
                fml = file.readline().strip()
                if not fml:
                    break
                print(fml)
                signal = self.test_factor(fml, end_date=date, prediction_mode=True)
                self.dump_factor(fml)
                self.dump_signal(signal)
        signal_path = 'F:/Documents/AutoFactoryData/Signal/{}-{}'.format(self.data.start_date, self.end_date)

        print('reading signal...')
        signals_dic = {}
        lst = os.listdir(signal_path)
        num = 0
        for s in lst:
            with open('{}/{}'.format(signal_path, s), 'rb') as file:
                tmp = pickle.load(file)
                signals_dic[num] = tmp
            num += 1
        self.back_tester.generate_signal(model, signals_dic, end_date=date)
        ll = self.back_tester.long_stock_predict(n=n)
        print(ll)

    def train(self):
        pass
