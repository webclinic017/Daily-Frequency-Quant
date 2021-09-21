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

结构说明：
1. AutoFactory类是核心，其初始化的时候将调用DataLoader读取数据，然后得到data_dic等保存在类属性中
2. AutoFormula文件夹下定义的都是和formula相关的类，因此所有和公式有关的方法，包括树的生成，解析，信号提取等，都应该在里面定义
3. Tester文件夹下定义的类都是和信号评价相关的类，因此所有和信号相关的方法都应该在里面定义，包括测试信号表现的
4. Model文件夹下定义的类都是和模型定义相关的，所有关于模型的代码，包括训练集生成等，都应该在里面定义
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
                 back_test_name='default', return_type='close_close_1', need_industry=False,
                 data_path=None, back_test_data_path=None, dump_factor_path=None):  # 暂时需要说明说否需要行业
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
        if data_path is None:
            data_path = 'F:/Documents/AutoFactoryData'
        self.data_path = data_path
        if back_test_data_path is None:
            back_test_data_path = 'F:/Documents/AutoFactoryData/BackTestData'
        self.back_test_data_path = back_test_data_path
        self.end_date = end_date
        self.dataloader = DataLoader(user_id, password, data_path=self.data_path,
                                     back_test_data_path=self.back_test_data_path)
        self.data = self.dataloader.get_matrix_data(start_date=start_date, back_test_name=back_test_name,
                                                    end_date=end_date, return_type=return_type,
                                                    need_industry=need_industry)

        if dump_signal_path is None:
            lst = os.listdir('F:/Documents/AutoFactoryData/Signal')
            if '{}-{}'.format(start_date, end_date) not in lst:
                os.makedirs('F:/Documents/AutoFactoryData/Signal/{}-{}'.format(start_date, end_date))
            dump_signal_path = 'F:/Documents/AutoFactoryData/Signal/{}-{}'.format(start_date, end_date)
        self.dump_signal_path = dump_signal_path
        if dump_factor_path is None:
            dump_factor_path = 'F:/Documents/AutoFactoryData/Factors'
        self.dump_factor_path = dump_factor_path
        self.back_tester = BackTester(data=self.data)  # 模拟交易回测
        self.autoformula = AutoFormula(start_date=start_date, end_date=end_date, data=self.data)
        self.dsc = DataSetConstructor(self.data, signal_path=self.dump_signal_path)

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
                         back_window=100, strategy='long_short', frequency='weekly', start_weekday=1,
                         zt_filter=True):  # 滚动回测
        """
        :param zt_filter: 是否过滤涨停
        :param model_name: 如果使用模型进行回测，说明模型名字
        :param start_date: 回测开始日期
        :param end_date: 回测结束日期
        :param n: 如果strategy是long_to_n，指定long多少只股票
        :param time_window: 同一个策略滚动使用多少次调仓，注意是调仓，如果是周频，time_window取1指的是每周调仓时都重新训练
        :param back_window: 训练模型时回溯多少个交易日，注意这里是交易日
        :param strategy: 根据信号构造仓位的方式
        :param frequency: 交易频率，daily是日频，weekly是周频
        :param start_weekday: 如果是周频，start_weekday字段指的是在周几买入
        :return:
        """
        if start_date is None:
            start_date = str(self.start_date)
        if end_date is None:
            end_date = str(self.end_date)
        start, end = self.data.get_real_date(start_date, end_date)
        assert start >= back_window  # 防止无法回溯
        model = Model()

        if start_weekday == 1:
            start_weekday = 4
        else:
            start_weekday -= 1  # 因为周一买入的股票用的是周五的信号
        total_signal = np.zeros(self.data.top.shape)  # 存储所有的信号
        pnl = []
        cumulated_pnl = []
        i = start
        self.dsc = DataSetConstructor(self.data)

        stride = 0  # 表示是否需要训练模型
        while i + time_window < end:
            if frequency == 'weekly' and self.data.position_date_dic[i].weekday() != start_weekday:
                i += 1
                continue  # 如果不是买入的日子，就略过
            s = i - back_window - 1
            e = i - 1
            s_date = str(self.data.position_date_dic[s])
            e_date = str(self.data.position_date_dic[e])
            s_forward = str(self.data.position_date_dic[i])
            e_forward = str(self.data.position_date_dic[i])
            print('testing {} to {}'.format(s_forward, e_forward))
            if stride == 0:
                x, y = self.dsc.construct(start_date=s_date, end_date=e_date)
                model.fit(x[:-6000, :], y[:-6000], x[-6000:, :], y[-6000:], model=model_name)
            stride += 1
            if stride == time_window:
                stride = 0

            signal = self.back_tester.generate_signal(model.model, self.dsc.signals_dic,
                                                      start_date=s_forward, end_date=e_forward)
            total_signal += signal
            if strategy == 'long_short':
                l = self.back_tester.long_short(start_date=s_forward, end_date=e_forward)
            if strategy == 'long_top_n':
                l = self.back_tester.long_top_n(start_date=s_forward, end_date=e_forward, n=n, zt_filter=zt_filter)
            if strategy == 'long':
                l = self.back_tester.long(start_date=s_forward, end_date=e_forward, n=0)
            pnl += self.back_tester.pnl  # pnl序列
            c_pnl = self.back_tester.cumulated_pnl.copy()
            if not cumulated_pnl:
                cumulated_pnl = c_pnl.copy()
            else:
                for j in range(len(c_pnl)):
                    c_pnl[j] += cumulated_pnl[-1]
                cumulated_pnl += c_pnl
            i += 1
        return pnl, cumulated_pnl, total_signal

    def test_signal(self, signal, n=0, strategy='long_short', zt_filter=True, position_mode='mean'):
        if strategy == 'long_short':
            self.back_tester.long_short(signal)
        if strategy == 'long_top_n':
            self.back_tester.long_top_n(signal=signal, n=n, zt_filter=zt_filter, position_mode=position_mode)
        print('mean pnl: {:.4f}, sharp_ratio: {:.4f}, max_dd: {:.4f}'.format(self.back_tester.mean_pnl * 100,
                                                                             self.back_tester.sharp_ratio,
                                                                             self.back_tester.max_dd * 100))

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

    def long_stock_predict(self, model_name=None, factor=None, date=None, n=1):  # 每日推荐股票多头
        """
        :param n: 推荐得分最高的n只股票
        :param model_name: 使用的模型，可以不使用模型
        :param factor: 使用的因子，可以是类型为字符串的绝对路径，也可以是列表
        :param date: 预测哪一天
        :return: 直接打印结果
        """
        if date is None:
            date = str(self.data.end_date)  # 这里之后的版本要修改成更加灵活的读写信号，例如每天自动增量更新
        if model_name is not None:
            print('reading model...')
            with open('F:/Documents/AutoFactoryData/Model/{}.pkl'.format(model_name), 'rb') as file:
                model = pickle.load(file)
        else:
            model = None
        print('getting signal...')
        if type(factor) == str:
            num = 0
            signals_dic = {}
            with open('F:/Documents/AutoFactoryData/Factors/{}.txt'.format(factor)) as file:
                while True:
                    fml = file.readline().strip()
                    if not fml:
                        break
                    signal = self.test_factor(fml, end_date=date, prediction_mode=True)
                    signals_dic[num] = signal
                    num += 1
            print('there are {} factors'.format(num))
        else:
            num = 0
            signals_dic = {}
            for fml in factor:
                signal = self.test_factor(fml, end_date=date, prediction_mode=True)
                signals_dic[num] = signal
                num += 1
            print('there are {} factors'.format(num))
        if model is None:
            self.back_tester.generate_signal(model, signals_dic, end_date=date)
            ll = self.back_tester.long_stock_predict(date=date, n=n)
        else:
            self.back_tester.generate_signal(model=None, signals_dic=signals_dic, end_date=date)
            ll = self.back_tester.long_stock_predict(date=date, n=n)
        print(ll)

    def train(self):
        pass
