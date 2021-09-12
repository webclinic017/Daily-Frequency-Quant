### Model

Model类定义了多种标准化以及结构化的模型训练框架，可以直接调用拟合。



### DataSetConstructor

DataSetConstructor类定义了生成训练集X和Y的函数



#### 开发日志

##### 2021-09-04

-- 定义了DataSetConstructor类，改成标准化的Data类输入

##### 2021-09-10

-- 新增：使用网格搜索方法对因子搜索系数，以优化指定区间内得分最高的n个股票的平均收益

##### 2021-09-11

-- 新增：lightgbm方法，以及Lasso和lightgbm做boosting



