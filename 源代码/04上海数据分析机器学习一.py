# -*- coding: utf-8 -*-
"""
Created on Fri 

@author: 
"""
##导入必要的库
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

###读取四个文件
data_analysis = pd.read_csv('data_analysis.csv').drop(['Unnamed: 0'],axis=1)
data_mining = pd.read_csv('data_mining.csv').drop(['Unnamed: 0'],axis=1)
machine_learning = pd.read_csv('machine_learning.csv').drop(['Unnamed: 0'],axis=1)
business_analysis = pd.read_csv('business_analysis.csv').drop(['Unnamed: 0'],axis=1)

##四个文件进行拼接
data=data_analysis.copy()
data = data.append(business_analysis, ignore_index=True)
data = data.append(machine_learning, ignore_index=True)
data = data.append(data_mining, ignore_index=True)

##查看是否存在缺失值
nulls = np.sum(data.isnull())
nullcols = nulls.loc[(nulls != 0)]
print(nullcols)


# 读取用户字典
jieba.load_userdict("userdict.txt")


# # 读取软技能字典
with open('软技能.txt',"r") as f:
    soft_skills = f.read()
skilllist = soft_skills.split('\n')


# # 读取硬技能字典
with open('硬技能.txt',"r") as f:
    hard_skills = f.read()

    
def match_soft(skills,templist):
    for item in skills:
        if item in skilllist:
            templist.append(item)
        
def match_hard(skill_list,templist):
    for skill in skill_list:
        if skill.capitalize() in hard_skills:
            templist.append(skill.capitalize())
                

pattern = '[a-zA-Z]+'
data_soft = []
data_hard = []

# 匹配软技能
for item in data['职位描述']:
    skill_list = jieba.cut(item)
    skill_list = list(set(skill_list)) 
    match_soft(skill_list,data_soft)
    
# 匹配硬技能
for item in data['职位描述']:
    skill_list = re.findall(pattern,item)
    skill_list = list(set(skill_list)) 
    match_hard(skill_list,data_hard)

##前10个最重要的软技能和硬技能
data_soft_counts = Series(data_soft).value_counts()[:10]
data_hard_counts = Series(data_hard).value_counts()[:10]



temp_data = {
    '软技能':data_soft_counts.index,
    'count':data_soft_counts.values
}

frame_soft = DataFrame(temp_data,index=[1,2,3,4,5,6,7,8,9,10])

temp_data = {
    '硬技能':data_hard_counts.index,
    'count':data_hard_counts.values
}

frame_hard = DataFrame(temp_data,index=[1,2,3,4,5,6,7,8,9,10])

data_copy = data.copy()

# 硬技能匹配函数
total_skill_list = []
def match(words):
    tmp_list = []
    for skill in frame_hard.硬技能.values:
        if skill in words:
            tmp_list.append(skill)
    total_skill_list.append(tmp_list)
    

for item in data_copy['职位描述'].values:
    english_words = re.findall(pattern,item)
    english_words = [item.capitalize() for item in english_words]
    english_words = list(set(english_words))
    match(english_words)

skill_series = Series(total_skill_list,index=data_copy.index)

for k in frame_hard.硬技能.values:
    data_copy[k] = 0
    
for i in skill_series.index:
    for skill in skill_series[i]:
        data_copy[skill][i] = 1
        
#data_copy.to_csv('data_with_skills.csv')


#机器学习部分

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

data_copy.salary.describe()

data_copy.salary.hist(bins=50, figsize=(8,5))
plt.show()

##看看salary和硬技能之间的关系
corr_matrix = data_copy.corr()
corr_matrix["salary"].sort_values(ascending=False)

data_analysis = data_copy.drop(['Keyword','职位描述','职位薪资','公司名称','职位名称'],axis=1)

##清洗地区数据
district = []
area = []

for item in data_analysis['地区']:
    dis_list = jieba.lcut(item)
    district.append(dis_list[2])

dis = Series(district)
data_analysis.insert(9,'区',dis)

##过滤没有地区的数据
data_analysis = data_analysis[data_analysis['区']!=' ']
data_analysis = data_analysis.drop(['地区'],axis=1)
data_analysis.columns

##查看缺失
nulls = np.sum(data_analysis.isnull())
nullcols = nulls.loc[(nulls != 0)]
print(nullcols)

##对类别变量独热
data_analysis = pd.get_dummies(data_analysis).reset_index(drop=True)

##划分训练集和测试集
train_set, test_set = train_test_split(data_analysis, test_size=0.2, random_state=42)
y_train=train_set['salary']
x_train=train_set.drop(['salary'],axis=1)
y_test=test_set['salary']
x_test=test_set.drop(['salary'],axis=1)
y_train=y_train.values.reshape(-1, 1)
y_test=y_test.values.reshape(-1, 1)             
                

##base model
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)         
##训练误差
salary_predictions = lin_reg.predict(x_train)
lin_mse = mean_squared_error(y_train, salary_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse       
##测试误差
y_test_pre = lin_reg.predict(x_test)
lin_mse = mean_squared_error(y_test, y_test_pre)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
##交叉验证表现
scores = cross_val_score(lin_reg, x_train, y_train,
                         scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-scores)
display_scores(lin_rmse_scores)


##决策树模型
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(x_train, y_train)
y_pred_tree = tree_reg.predict(x_train)
##训练误差
tree_mse = mean_squared_error(y_train, y_pred_tree)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
##测试误差
y_test_pre = tree_reg.predict(x_test)
tree_mse = mean_squared_error(y_test, y_test_pre)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

###随机森林模型
forest_reg = RandomForestRegressor(random_state=52)
forest_reg.fit(x_train, y_train)
##训练误差
y_pred_rf = forest_reg.predict(x_train)
forest_mse = mean_squared_error(y_train, y_pred_rf)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
##测试误差
y_test_pre = forest_reg.predict(x_test)
forest_mse = mean_squared_error(y_test,y_test_pre )
forest_rmse = np.sqrt(forest_mse)
forest_rmse
##交叉验证表现
scores = cross_val_score(forest_reg, x_train, y_train.ravel(),
                         scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)
                
##k近邻回归
k = 5
knn_reg = KNeighborsRegressor(k)
knn_reg.fit(x_train, y_train)
#训练误差
y_pred_knn = knn_reg.predict(x_train)
knn_mse = mean_squared_error(y_train, y_pred_knn)
knn_rmse = np.sqrt(knn_mse)
knn_rmse 
##测试误差
y_test_pre = knn_reg.predict(x_test)
knn_mse = mean_squared_error(y_test, y_test_pre)
knn_rmse = np.sqrt(knn_mse)
knn_rmse
##交叉验证表现
scores = cross_val_score(knn_reg, x_train, y_train,
                         scoring="neg_mean_squared_error", cv=10)
knn_rmse_scores = np.sqrt(-scores)
display_scores(knn_rmse_scores)        
      

##adaboost
Adaboost_reg = AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
Adaboost_reg.fit(x_train, y_train)
##训练误差
y_pred_Ada = knn_reg.predict(x_train)
Ada_mse = mean_squared_error(y_train, y_pred_Ada)
Ada_rmse = np.sqrt(Ada_mse) 
##测试误差
y_test_pre = Adaboost_reg.predict(x_test)
Adaboost_mse = mean_squared_error(y_test, y_test_pre)
Adaboost_rmse = np.sqrt(Adaboost_mse)
Adaboost_rmse 
##交叉验证表现
scores = cross_val_score(Adaboost_reg, x_train, y_train.ravel(),
                         scoring="neg_mean_squared_error", cv=10)
Adaboost_rmse_scores = np.sqrt(-scores)
display_scores(Adaboost_rmse_scores)                   

##grbt
grbt_reg = GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
grbt_reg.fit(x_train, y_train)
##训练误差
y_pred_grbt = grbt_reg.predict(x_train)
grbt_mse = mean_squared_error(y_train, y_pred_grbt)
grbt_rmse = np.sqrt(grbt_mse)
grbt_rmse
##测试误差
y_test_pre = grbt_reg.predict(x_test)
grbt_mse = mean_squared_error(y_test, y_test_pre)
grbt_rmse = np.sqrt(grbt_mse)
grbt_rmse
##交叉验证表现
scores = cross_val_score(grbt_reg, x_train, y_train.ravel(),
                         scoring="neg_mean_squared_error", cv=10)
grbt_rmse_scores = np.sqrt(-scores)
display_scores(grbt_rmse_scores)

##bagging
bagging_reg = BaggingRegressor()
bagging_reg.fit(x_train, y_train)
##训练误差
y_pred_bag = bagging_reg.predict(x_train)
bagging_mse = mean_squared_error(y_train, y_pred_bag)
bagging_rmse = np.sqrt(bagging_mse)
bagging_rmse
##测试误差
y_test_pre = bagging_reg.predict(x_test)
bagging_mse = mean_squared_error(y_test, y_test_pre)
bagging_rmse = np.sqrt(bagging_mse)
bagging_rmse
##交叉验证表现
scores = cross_val_score(bagging_reg, x_train, y_train,
                         scoring="neg_mean_squared_error", cv=10)
bagging_rmse_scores = np.sqrt(-scores)
display_scores(bagging_rmse_scores)

##各个模型表现
model_list = [lin_rmse,tree_rmse,forest_rmse,knn_rmse,Adaboost_rmse,grbt_rmse,bagging_rmse]
model_name = ['linear','tree','forest','knn','Adaboost','grbt','bagging']
i = 0
for model in model_list:
    print(model_name[i],'在测试集上的误差表现为：',model)
    i+=1
    

##选取grbt进行调参
param_grid = [   
    {'n_estimators': [50,100,150], 'max_features': [2, 4, 6, 8],'max_depth':[3,5,7]}
  ]

grbt_reg = GradientBoostingRegressor()
grid_search = GridSearchCV(grbt_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(x_train, y_train.ravel())

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
   
grid_search.best_params_

feature_importances = grid_search.best_estimator_.feature_importances_

# 变量重要性排序
attributes = list(x_train.columns)
sorted(zip(feature_importances, attributes), reverse=True)

##最终模型
final_model = grid_search.best_estimator_
##测试误差
final_predictions = final_model.predict(x_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse

##预测
def func_params(templist):
    tmp=np.zeros((1,len(x_test.columns)))
    df = pd.DataFrame(tmp,columns = list(x_test.columns))
    scale,degree,exp,skills = templist
    scalepara = "公司规模_"+scale
    degreepara = "学历要求_"+degree
    exppara = "工作经验_"+exp
    df[scalepara]=1
    df[degreepara]=1
    df[exppara]=1  
    for x in skills:
        df[x]=1
    temp = np.array(df, dtype = float).reshape(1, -1)
    return temp

def predict(templist):
    X_predict = func_params(templist)
    Y_predict = final_model.predict(X_predict)
    print('预测薪资为：',Y_predict[0],'k')


templist = ['10000人以上','本科','1-3年',['Sql','Python','Excel']]
predict(templist)















