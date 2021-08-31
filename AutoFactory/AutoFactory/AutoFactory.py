# Copyright (c) 2021 Dai HBG

"""
AutoFactory类集成了给定因子表达式自动回测并评价因子，自动搜索因子的方法
"""


class AutoFactory:
    def __init__(self):
        self.back_tester = BackTester()

    def back_test(self, formula, start_date, end_date):
        """
        :param formula: 回测的公式
        :param start_date: 回测开始日期
        :param end_date: 回测结束日期
        :return:
        """
        self.back_tester.test(formula, start_date, end_date)
