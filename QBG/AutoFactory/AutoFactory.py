# Copyright (c) 2021 Dai HBG

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
from Model import Model
from DataSetConstructor import DataSetConstructor

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

"""
开发日志
2021-09-06
-- 更新：使用预测函数时所需的信号放在内存中，不需要重复读取，除非以后有更加复杂的模型
2021-09-11
-- 更新：新增滚动回测方法，测试模型的长期稳健性。默认可以回溯100天滚动5天预测
"""


class AutoFactory:
    def __init__(self, user_id, password, start_date, end_date, dump_signal_path=None,
                 back_test_name='default', return_type='close_close_1'):
        """
        :param user_id: 登录聚宽的用户id
        :param password: 登录密码
        :param start_date: 总体的开始日期
        :param end_date: 总体的结束日期
        :param dump_signal_path: dump信号矩阵路径
        :param back_test_name: 回测名称
        :param return_type: 收益率预测形式，默认是收盘价到收盘价，意味着日度调仓
        """
        self.start_date = start_date
        self.end_date = end_date
        self.dataloader = DataLoader(user_id, password)
        self.data = self.dataloader.get_matrix_data(start_date=start_date, back_test_name=back_test_name,
                                                    end_date=end_date, return_type=return_type)

        if dump_signal_path is None:
            lst = os.listdir('F:/Documents/AutoFactoryData/Signal')
            if '{}-{}'.format(start_date, end_date) not in lst:
                os.makedirs('F:/Documents/AutoFactoryData/Signal/{}-{}'.format(start_date, end_date))
            dump_signal_path = 'F:/Documents/AutoFactoryData/Signal/{}-{}'.format(start_date, end_date)
        self.dump_signal_path = dump_signal_path
        self.back_tester = BackTester(data=self.data)  # 模拟交易回测
        self.autoformula = AutoFormula(start_date=start_date, end_date=end_date, data=self.data)
        self.dsc = DataSetConstructor(self.data)
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

    def rolling_backtest(self, model_name='lgbm', start_date=None, end_date=None, n=3, time_window=5,
                         back_window=100, strategy='long_short'):  # 滚动回测
        if start_date is None:
            start_date = str(self.start_date)
        if end_date is None:
            end_date = str(self.end_date)
        start, end = self.data.get_real_date(start_date, end_date)
        assert start >= back_window  # 防止无法回溯
        model = Model()
        pnl = []
        cumulated_pnl = []
        i = start
        self.dsc = DataSetConstructor(self.data)
        while i + time_window < end:
            s = i - back_window - 1
            e = i - 1
            s_date = str(self.data.position_date_dic[s])
            e_date = str(self.data.position_date_dic[e])
            s_forward = str(self.data.position_date_dic[i])
            e_forward = str(self.data.position_date_dic[i + time_window])
            print('testing {} to {}'.format(s_forward, e_forward))
            x, y = self.dsc.construct(start_date=s_date, end_date=e_date)
            model.fit(x[:-6000, :], y[:-6000], x[-6000:, :], y[-6000:], model=model_name)
            signal = self.back_tester.generate_signal(model.model, self.dsc.signals_dic,
                                                      start_date=s_forward, end_date=e_forward)
            if strategy == 'long_short':
                l = self.back_tester.long_short(start_date=s_forward, end_date=e_forward)
            if strategy == 'long_top_n':
                l = self.back_tester.long_top_n(start_date=s_forward, end_date=e_forward, n=n)
            if strategy == 'long':
                l = self.back_tester.long(start_date=s_forward, end_date=e_forward, n=0)
            pnl += self.back_tester.pnl
            c_pnl = self.back_tester.cumulated_pnl.copy()
            if not cumulated_pnl:
                cumulated_pnl = c_pnl.copy()
            else:
                for j in range(len(c_pnl)):
                    c_pnl[j] += cumulated_pnl[-1]
                cumulated_pnl += c_pnl
            i += time_window
        return pnl, cumulated_pnl

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

    def long_stock_predict(self, model_name, factor, date=None, n=1):  # 每日推荐股票多头
        """
        :param n: 推荐得分最高的n只股票
        :param model_name: 使用的模型
        :param factor: 使用的因子
        :param date: 预测哪一天
        :return: 直接打印结果
        """
        if date is None:
            date = str(self.data.end_date)  # 这里之后的版本要修改成更加灵活的读写信号，例如每天自动增量更新
        print('reading model...')
        with open('F:/Documents/AutoFactoryData/Model/{}.pkl'.format(model_name), 'rb') as file:
            model = pickle.load(file)
        print('getting signal...')
        num = 0
        signals_dic = {}
        with open('F:/Documents/AutoFactoryData/Factors/{}.txt'.format(factor)) as file:
            while True:
                fml = file.readline().strip()
                if not fml:
                    break
                # print(fml)
                signal = self.test_factor(fml, end_date=date, prediction_mode=True)
                signals_dic[num] = signal
                num += 1
                # self.dump_factor(fml)
                # self.dump_signal(signal)
        # signal_path = 'F:/Documents/AutoFactoryData/Signal/{}-{}'.format(self.data.start_date, self.end_date)
        print('there are {} factors'.format(num))
        self.back_tester.generate_signal(model, signals_dic, end_date=date)
        ll = self.back_tester.long_stock_predict(date=date, n=n)
        print(ll)

    def train(self):
        pass