# Copyright (c) 2021 Dai HBG

"""
Model是若干个定义了多种标准化以及结构化的模型框架，可以直接调用用于模型拟合
"""
from sklearn import linear_model
import numpy as np
import pickle


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
            pass
        if model == 'Lasso_lgbm_boosting':
            pass
        if model == 'Lasso':
            pass

    def model_fit(self, model=None):  # 传入自定义模型进行训练
        if model is None:
            self.model = linear_model.Lasso(alpha=6e-4)
        else:
            self.model = model
        pass

    def dump_model(self, model_name):  # 保存模型
        with open('F:/Documents/AutoFactoryData/Model/{}.pkl'.format(model_name), 'wb') as file:
            pickle.dump(self.model, file)
