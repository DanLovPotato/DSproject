# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 10:01:55 2018

@author: hp
"""
##导入必要的库
import pandas as pd
from pandas import Series
import jieba
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def prepare_model():
    data_copy = pd.read_csv('data_with_skills.csv',encoding='gbk')
    data_copy = data_copy.drop(['Unnamed: 0'],axis=1)
    #处理salary
    def average(job_salary):
        pattern = re.compile('\d+')
        salary = job_salary
        try:
            res = re.findall(pattern, salary)
            avg_salary = 0
            sum = 0
            for i in res:
                a = int(i)
                sum = sum + a
                avg_salary = sum / 2
        except Exception:
            avg_salary = 0
        return avg_salary
    
    salary_list = []
    
    for i in range(0,data_copy.shape[0]):
        avg_sal = average(data_copy['职位薪资'][i])
        salary_list.append(avg_sal)
        
    sal = Series(salary_list)
    
    data_copy.insert(9,'salary',sal)
    
    data_analysis = data_copy.drop(['Keyword','职位描述','职位薪资','公司名称','职位名称'],axis=1)
    
    ##清洗地区数据
    district = []
    for item in data_analysis['地区']:
        dis_list = jieba.lcut(item)
        district.append(dis_list[2])
    
    dis = Series(district)
    data_analysis.insert(9,'区',dis)
    
    
    ##过滤没有地区的数据
    data_analysis = data_analysis[data_analysis['区']!=' ']
    data_analysis = data_analysis.drop(['地区'],axis=1)
    
    ##对类别变量独热
    data_analysis = pd.get_dummies(data_analysis).reset_index(drop=True)
    
    ##划分训练集和测试集
    train_set, test_set = train_test_split(data_analysis, test_size=0.2, random_state=42)
    y_train=train_set['salary']
    x_train=train_set.drop(['salary'],axis=1)
    columns1=x_train.columns
    ##最终模型
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    #stack
    lasso = Lasso(alpha=0.02)
    grbt = GradientBoostingRegressor(max_depth = 7, max_features = 2, n_estimators = 100)
    rf = RandomForestRegressor(n_estimators = 150)
    xgb = XGBRegressor(max_depth = 5, min_child_weight =3, reg_alpha = 0.01)
    stack_gen = StackingCVRegressor(regressors=(lasso, grbt, 
                                                rf,xgb), 
                                   meta_regressor=LinearRegression()
                                   )
       
    stackX = x_train
    stacky = y_train
    final_model = stack_gen.fit(stackX, stacky)
    return final_model,columns1

##预测
def func_params(predictors,columns1):
    tmp = np.zeros((1,len(columns1)))
    df = pd.DataFrame(tmp,columns = list(columns1))
    scale = predictors['公司规模'][0]
    degree = predictors['学历要求'][0]
    exp = predictors['工作经验'][0]
    scalepara = "公司规模_"+scale
    degreepara = "学历要求_"+degree
    exppara = "工作经验_"+exp
    df[scalepara]=1
    df[degreepara]=1
    df[exppara]=1  
    for x in ['Python', 'Sql', 'R', 'Java', 'Hadoop', 'C', 'Spark', 'Excel',
       'Hive', 'Sas']:
        if x in predictors:
            df[x]=1
    temp = np.array(df, dtype = float).reshape(1, -1)
    return temp

def predict(predictors):
    final_model, columns1 = prepare_model()
    X_predict = func_params(predictors,columns1)
    Y_predict = final_model.predict(X_predict)
    return Y_predict[0]


#templist = '10000人以上,本科,1-3年,Sql,Python,Excel'
#predict(templist)















