# -*- coding: utf-8 -*-
"""
Created on Sat

@author: 
"""

import pandas as pd
from pandas import Series
import jieba
import numpy as np
import re
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor              
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.stats import norm
from sklearn.linear_model import Lasso
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor

data_copy = pd.read_csv('data_with_skills.csv',encoding='gbk')
data_copy = data_copy.drop(['Unnamed: 0'],axis=1)
data_copy.shape

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
area = []

for item in data_analysis['地区']:
    dis_list = jieba.lcut(item)
    district.append(dis_list[2])

dis = Series(district)
data_analysis.insert(9,'区',dis)
dis_count = dis.value_counts()


##过滤没有地区的数据
data_analysis = data_analysis[data_analysis['区']!=' ']
data_analysis = data_analysis.drop(['地区'],axis=1)
data_analysis.columns

##查看缺失
nulls = np.sum(data_analysis.isnull())
nullcols = nulls.loc[(nulls != 0)]
print(nullcols)

##对类别变量独热

data_analysis1 = pd.get_dummies(data_analysis).reset_index(drop=True)

##划分训练集和测试集
train_set, test_set = train_test_split(data_analysis1, test_size=0.2, random_state=42)
y_train = np.array(train_set['salary'])
x_train=train_set.drop(['salary'],axis=1)
y_test = np.array(test_set['salary'])
x_test=test_set.drop(['salary'],axis=1)
 
x_train1 = x_train.copy()
x_test1 = x_test.copy()


xg_reg =  XGBRegressor(n_estimators = 100, max_depth = 5, min_child_weight = 3)
xg_reg.fit(x_train,y_train)

y_test_pred = xg_reg.predict(x_test)
test_error = np.sqrt(mean_squared_error(y_test, y_test_pred))
##7.3538205489644835

importances = xg_reg.feature_importances_
feature_importances = pd.DataFrame(importances,index = x_train1.columns)
feature_importances_del = feature_importances[feature_importances[0]==0]

##考虑删除特征重要性为0的因素，一个一个删除后效果没有多大变化
for x in feature_importances_del.index:
    x_train_tmp = x_train.drop([x],axis=1)
    x_test_tmp = x_test.drop([x],axis=1)
    xg_reg =  XGBRegressor(n_estimators = 100, max_depth = 5, min_child_weight = 3).fit(x_train_tmp,y_train)
    y_test_pred = xg_reg.predict(x_test_tmp)    
    test_error = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(x,test_error)

#学历要求_中专/中技 7.35382054896
#学历要求_高中 7.35382054896
#融资情况_10000人以上 7.35382054896
#融资情况_500-999人 7.35382054896
#区_奉贤区 7.35382054896
x_train_tmp = x_train.drop(list(feature_importances_del.index),axis=1)
x_test_tmp = x_test.drop(list(feature_importances_del.index),axis=1)
xg_reg =  XGBRegressor(n_estimators = 100, max_depth = 5, min_child_weight = 3).fit(x_train_tmp,y_train)
y_test_pred = xg_reg.predict(x_test_tmp)    
test_error = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(test_error)

##增加rank
columns = data_analysis.columns
company_features = ['公司规模','区','融资情况']
employee_features = ['工作经验','学历要求']
cat_features = company_features+employee_features


for x in cat_features:
    data_analysis_tmp = data_analysis.copy()
    rank_df = data_analysis_tmp.loc[:,[x,'salary']].groupby(x).mean()
    rank_df.columns = [x+'_rank']
    rank_df.reset_index(inplace = True)
    data_analysis_new = pd.merge(data_analysis_tmp,rank_df,on = x,how='left')
    data_analysis_new = pd.get_dummies(data_analysis_new).reset_index(drop=True)
    data_analysis_new = data_analysis_new.drop(['salary']+list(feature_importances_del.index),axis=1)
    x_train_tmp = data_analysis_new.loc[x_train1.index,:]
    x_test_tmp = data_analysis_new.loc[x_test1.index,:]
    xg_reg =  XGBRegressor(n_estimators = 100, max_depth = 5, min_child_weight = 3).fit(x_train_tmp,y_train)
    y_test_pred = xg_reg.predict(x_test_tmp)    
    test_error = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(x,test_error)
    
#公司规模 7.2957937905
#区 7.19963162164
#融资情况 7.16665055616
#工作经验 7.22908525677
#学历要求 7.30895389543
    
##考虑一起加进去
data_analysis_new = data_analysis.copy()
for x in cat_features:
    rank_df = data_analysis_tmp.loc[:,[x,'salary']].groupby(x).mean()
    rank_df.columns = [x+'_rank']
    rank_df.reset_index(inplace = True)
    data_analysis_new = pd.merge(data_analysis_new,rank_df,on = x,how='left')
data_analysis_new = pd.get_dummies(data_analysis_new).reset_index(drop=True)
data_analysis_new = data_analysis_new.drop(['salary']+list(feature_importances_del.index),axis=1)
x_train_tmp = data_analysis_new.loc[x_train1.index,:]
x_test_tmp = data_analysis_new.loc[x_test1.index,:]
xg_reg =  XGBRegressor(n_estimators = 100, max_depth = 5, min_child_weight = 3).fit(x_train_tmp,y_train)
y_test_pred = xg_reg.predict(x_test_tmp)    
test_error = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(test_error)
pd.DataFrame(xg_reg.feature_importances_,index = x_train_tmp.columns).sort_values(by = 0,axis = 0,ascending = True)
#一起加进去，7.2571244082198989

new = ['融资情况','区','工作经验','公司规模','学历要求']
for x in new[::-1]:
    x = '学历要求'
    x_train_tmp1 = x_train_tmp.drop([x+'_rank'],axis=1)
    x_test_tmp1 = x_test_tmp.drop([x+'_rank'],axis=1)
    xg_reg =  XGBRegressor(n_estimators = 100, max_depth = 5, min_child_weight = 3).fit(x_train_tmp1,y_train)
    y_test_pred = xg_reg.predict(x_test_tmp1)    
    test_error1 = np.sqrt(mean_squared_error(y_test, y_test_pred))
    if test_error>=test_error1:
        print(x)
        x_train_tmp = x_train_tmp.drop([x+'_rank'],axis=1)
        x_test_tmp = x_test_tmp.drop([x+'_rank'],axis=1)
        test_error=test_error1

##从后退法来看，最好把所有的都保留

##加聚类标签
##个人因素聚类标签
employee_features
cluster1 = data_analysis[employee_features+['salary']]
cluster1 = pd.get_dummies(cluster1).reset_index(drop=True)
cluster1_columns = cluster1.columns

from sklearn.preprocessing import MinMaxScaler,StandardScaler
#mm_scaler=MinMaxScaler()
mm_scaler = StandardScaler()
cluster1 = mm_scaler.fit_transform(cluster1)
from sklearn.cluster import DBSCAN
cls_model=DBSCAN(eps=0.1, min_samples=3,n_jobs=-1).fit(cluster1)
cls_model.labels_
data_analysis_new.loc[:,'cls1'] = cls_model.labels_
x_train_tmp = data_analysis_new.loc[x_train1.index,:]
x_test_tmp = data_analysis_new.loc[x_test1.index,:]
xg_reg =  XGBRegressor(n_estimators = 100, max_depth = 5, min_child_weight = 3).fit(x_train_tmp,y_train)
y_test_pred = xg_reg.predict(x_test_tmp)    
test_error = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(test_error)
##7.30503557882 MinMaxScaler
##4.18407616321 StandardScaler

##公司因素聚类标签
company_features 
cluster2 = data_analysis[company_features +['salary']]
cluster2 = pd.get_dummies(cluster2).reset_index(drop=True)
cluster2_columns = cluster2.columns

from sklearn.preprocessing import MinMaxScaler,StandardScaler
mm_scaler=MinMaxScaler()
#mm_scaler = StandardScaler()
cluster2 = mm_scaler.fit_transform(cluster2)
from sklearn.cluster import DBSCAN
cls_model=DBSCAN(eps=0.1, min_samples=3,n_jobs=-1).fit(cluster2)
cls_model.labels_
data_analysis_new.loc[:,'cls2'] = cls_model.labels_
x_train_tmp = data_analysis_new.loc[x_train1.index,:]
x_test_tmp = data_analysis_new.loc[x_test1.index,:]
xg_reg =  XGBRegressor(n_estimators = 100, max_depth = 5, min_child_weight = 3).fit(x_train_tmp,y_train)
y_test_pred = xg_reg.predict(x_test_tmp)    
test_error = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(test_error)
#3.61681988681 MinMaxScaler
#4.35501811209 StandardScaler

##技能聚类标签
skills = ['Python', 'Sql', 'R', 'Java', 'Hadoop', 'C', 'Spark', 'Excel', 'Hive', 'Sas']
cluster3 = data_analysis[skills +['salary']]
cluster3 = pd.get_dummies(cluster3).reset_index(drop=True)
cluster3_columns = cluster3.columns

from sklearn.preprocessing import MinMaxScaler,StandardScaler
#mm_scaler=MinMaxScaler()
mm_scaler = StandardScaler()
cluster3 = mm_scaler.fit_transform(cluster3)
from sklearn.cluster import DBSCAN
cls_model=DBSCAN(eps=0.1, min_samples=3,n_jobs=-1).fit(cluster3)
cls_model.labels_
data_analysis_new.loc[:,'cls3'] = cls_model.labels_
x_train_tmp = data_analysis_new.loc[x_train1.index,:]
x_test_tmp = data_analysis_new.loc[x_test1.index,:]
xg_reg =  XGBRegressor(n_estimators = 100, max_depth = 5, min_child_weight = 3).fit(x_train_tmp,y_train)
y_test_pred = xg_reg.predict(x_test_tmp)    
test_error = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(test_error)
#3.96577311865 MinMaxScaler
#3.70012354984 StandardScaler 

data_analysis_new = data_analysis_new.drop(['cls3'],axis=1)
x_train = data_analysis_new.loc[x_train1.index,:]
x_test = data_analysis_new.loc[x_test1.index,:]
##开始调参
##lasso调参
param_grid = [   
    {'alpha':[0.05,0.06,0.1,0.11,0.12,0.13,0.15,0.17,0.2,0.4] }
  ]

lasso_reg = Lasso()
grid_search = GridSearchCV(lasso_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(x_train, y_train.ravel())

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
   
grid_search.best_params_
##最好 {'alpha': 0.1}

##随机森林调参
param_grid = [   
    {'n_estimators':[50,100,150,200]}
  ]

rf_reg =  RandomForestRegressor()
grid_search = GridSearchCV(rf_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(x_train, y_train.ravel())

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
   
grid_search.best_params_
# {'n_estimators': 150}


param_grid = [   
    {'max_depth':[20,25,30,35,40], 'min_samples_split':[2,3,4]}
  ]
rf_reg =  RandomForestRegressor(n_estimators=150)
grid_search = GridSearchCV(rf_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(x_train, y_train.ravel())

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
   
grid_search.best_params_
##{'max_depth': 25, 'min_samples_split': 2}
##xgboost调参
param_grid = [   
    {'n_estimators':[898,900,901]}
  ]


xg_reg =  XGBRegressor()
grid_search = GridSearchCV(xg_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(x_train, y_train.ravel())

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
   
grid_search.best_params_
#{'n_estimators': 900}
param_grid = [   
    {'max_depth':range(3,10,2),
     'min_child_weight':range(1,6,2),
   'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}
  ]

xg_reg =  XGBRegressor(n_estimators = 900)
grid_search = GridSearchCV(xg_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(x_train, y_train.ravel())

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
   
grid_search.best_params_
##{'max_depth': 3, 'min_child_weight': 5, 'reg_alpha': 1e-05}

##grbt调参
param_grid = [   
    {'n_estimators': [623,624,625]}
  ]

grbt_reg = GradientBoostingRegressor()
grid_search = GridSearchCV(grbt_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(x_train, y_train.ravel())

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
   
grid_search.best_params_
# {'n_estimators': 624}
param_grid = [   
    { 'min_samples_split':[499,500,501],'max_depth':[19,20,21]}
  ]

grbt_reg = GradientBoostingRegressor(n_estimators=624)
grid_search = GridSearchCV(grbt_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(x_train, y_train.ravel())

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
   
grid_search.best_params_
# {'max_depth': 20, 'min_samples_split': 500}



x_train1 = x_train.copy()
x_test1 = x_test.copy()
x_train=np.array(x_train)
x_test=np.array(x_test)


#stack
lasso = Lasso(alpha=0.1)
grbt = GradientBoostingRegressor(min_samples_split = 500, max_depth = 20, n_estimators = 624)
rf = RandomForestRegressor(n_estimators = 150,max_depth = 25, min_samples_split = 2)
xgb = XGBRegressor(n_estimators = 900,max_depth = 3, min_child_weight = 5, reg_alpha = 1e-05)
stack_gen = StackingCVRegressor(regressors=(lasso, grbt, 
                                            rf,xgb), 
                               meta_regressor=LinearRegression()
                               )


stackX = x_train
stacky = y_train
stack_gen_model = stack_gen.fit(stackX, stacky)

y_pre = stack_gen_model.predict(x_test)
final_mse = mean_squared_error(y_test, y_pre)
np.sqrt(final_mse)
#3.5625312428187335





























