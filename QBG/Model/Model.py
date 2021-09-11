# Copyright (c) 2021 Dai HBG

"""
Model是若干个定义了多种标准化以及结构化的模型框架，可以直接调用用于模型拟合

开发日志：
2021-09-10
-- 新增：使用网格搜索方法对因子搜索系数，以优化指定区间内得分最高的n个股票的平均收益
"""
import sys
from sklearn import linear_model
from copy import deepcopy
import numpy as np
import pickle
sys.path.append('C:/Users/Administrator/Desktop/Daily-Frequency-Quant/AutoFactory/Tester/')
from AutoTester import AutoTester
import lightgbm as lgb


class MyLinearModel:
    def __init__(self):
        self.coef_ = None  # 系数
        self.tester = AutoTester()

    def predict(self, x):
        assert len(x) >= 1
        return np.sum(x * self.coef_, axis=1)

    def gen_all_coef(self, n):  # 生成全排列
        if n == 1:
            return [[i * 0.1] for i in range(-10, 11)]
        else:
            a = self.gen_all_coef(n - 1)
            r = []
            for i in range(-10, 11):
                tmp = deepcopy(a)
                for k in tmp:
                    k.append(i)
                    r.append(k)
            return r

    def fit(self, signals_dic, ret, top, n=5):
        """
        :param signals_dic: 信号字典
        :param ret: 收益率矩阵
        :param top: 股票池
        :param n: 优化的平均收益
        :return: 直接优化系数
        """
        num = len(signals_dic)
        opt_ret = -0.1
        opt_coef = np.zeros(num)
        coefs = self.gen_all_coef(num)
        count = 0
        for coef in coefs:
            count += 1
            if count % 100 == 0:
                print('{} epochs done'.format(count))
            signal = np.sum([coef[i] * signals_dic[i] for i in range(num)], axis=0)
            stats = self.tester.test(signal, ret, top)
            if np.mean(stats.top_n_ret[5]) > opt_ret:
                print('now the opt_ret is {:.4f}'.format(np.mean(stats.top_n_ret[5])*100))
                opt_ret = np.mean(stats.top_n_ret[5])
                opt_coef = np.array(coef)
        self.coef_ = opt_coef


class Model:
    def __init__(self):
        self.model = None

    def fit(self, x_train, y_train, x_test=None, y_test=None, model=None, param=None):
        """
        :param x_train: 训练集x
        :param y_train: 训练集y
        :param x_test: 测试集x
        :param y_test: 测试集y
        :param model: 结构化模型名字
        :param param: 该模型对应的参数
        :return:
        """
        if model is None or model == 'Lasso':  # 默认使用LASSO
            self.model = linear_model.Lasso(alpha=5e-4)
            print('there are {} factors'.format(x_train.shape[1]))
            self.model.fit(x_train, y_train)
            print('{} factors have been selected'.format(np.sum(self.model.coef_ != 0)))
            print('training corr is {:.4f}'.format(np.corrcoef(y_train, self.model.predict(x_train))[0, 1]))
            if x_test is not None:
                print('testing corr is {:.4f}'.format(np.corrcoef(y_test, self.model.predict(x_test))[0, 1]))
        if model == 'lgbm':
            params = {'num_leaves': 20, 'min_data_in_leaf': 50, 'objective': 'regression', 'max_depth': 6,
                      'learning_rate': 0.05, "min_sum_hessian_in_leaf": 6,
                      "boosting": "gbdt", "feature_fraction": 0.9, "bagging_freq": 1, "bagging_fraction": 0.7,
                      "bagging_seed": 11, "lambda_l1": 2, "verbosity": 1, "nthread": -1,
                      'metric': 'mae', "random_state": 2019}  # 'device': 'gpu'}
            num_round = 100
            trn_data = lgb.Dataset(x_train, label=y_train)
            val_data = lgb.Dataset(x_test, label=y_test)
            self.model = lgb.train(params, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=20)
        if model == 'Lasso_lgbm_boosting':
            pass
        if model == 'Lasso':
            pass

    def fit_top_n(self, signals_dic, ret, top, n=5):
        """
        :param ret:
        :param signals_dic:
        :param n: 优化前多少只股票的平均收益
        :return:
        """
        self.model = MyLinearModel()
        self.model.fit(signals_dic, ret, top, n)

    def model_fit(self, model=None):  # 传入自定义模型进行训练
        if model is None:
            self.model = linear_model.Lasso(alpha=6e-4)
        else:
            self.model = model
        pass

    def dump_model(self, model_name):  # 保存模型
        with open('F:/Documents/AutoFactoryData/Model/{}.pkl'.format(model_name), 'wb') as file:
            pickle.dump(self.model, file)
