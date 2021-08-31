# Copyright (c) 2021 Dai HBG

"""
该代码定义的SignalGenerator类将解析一个公式，然后递归地计算signal值
v1.0
默认所有操作将nan替换成0
"""

import numba as nb

class SignalGenerator:
    def __init__(self):
        self.operation_dic = {}

    def get_operation(self):
        """
        定义算子
        :return: 形状一致的矩阵，也就是signal
        """

        """
        双目运算符
        """
        def add(a, b):
            return a + b
        self.operation_dic['add'] = add

        def minus(a, b):
            return a - b
        self.operation_dic['minus'] = minus

        def prod(a, b):
            c = a * b
            c[np.isnan(c)] = 0
            c[np.isinf(c)] = 0
            return a * b
        self.operation_dic['prod'] = prod

        def div(a, b):
            c = a / b
            c[np.isnan(c)] = 0
            c[np.isinf(c)] = 0
        self.operation_dic['div'] = div

        """
        单目运算符
        """

        @nb.jit()
        def csrank(a):
            b = a.copy()  # 测试用，可以不用复制
            for i in range(len(a)):
                n = np.sum(a[i] != 0)
                if n == 0:
                    continue
                tmp = a[i][a[i] != 0].copy()
                pos = tmp.argsort()
                for j in range(len(tmp)):
                    tmp[pos[j]] = j
                tmp /= (n - 1)
                b[i][b[i] != 0] = tmp
            return b
        self.operation_dic['csrank'] = csrank




