# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 19:36:37 2018

@author: hp
"""


from flask import Flask, render_template, request, redirect, make_response, send_file
import os
import pandas as pd
import numpy as np
from io import BytesIO
from flask_bootstrap import Bootstrap
import models as algorithms
from pandas import Series
import jieba
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor              
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor


app = Flask(__name__)

bootstrap = Bootstrap(app)

def datasetList():
    datasets = [x.split('.')[0] for f in ['datasets'] for x in os.listdir(f)]
    extensions = [x.split('.')[1] for f in ['datasets'] for x in os.listdir(f)]
    folders = [f for f in ['datasets'] for x in os.listdir(f)]
    return datasets, extensions, folders

#Load columns of the dataset
def loadColumns(dataset):
    datasets, extensions, folders = datasetList()
    if dataset in datasets:
        extension = extensions[datasets.index(dataset)]
        if extension == 'txt':
            df = pd.read_table(os.path.join(folders[datasets.index(dataset)], dataset + '.txt'))
        elif extension == 'csv':
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'),encoding='gbk').drop(['Unnamed: 0','职位描述'],axis=1)
        return df.columns
#Load Dataset    
def loadDataset(dataset):
    datasets, extensions, folders = datasetList()
    if dataset in datasets:
        extension = extensions[datasets.index(dataset)]
        if extension == 'txt':
            df = pd.read_table(os.path.join(folders[datasets.index(dataset)], dataset + '.txt'))
        elif extension == 'csv':
            df = pd.read_csv(os.path.join(folders[datasets.index(dataset)], dataset + '.csv'),encoding='gbk').drop(['Unnamed: 0','职位描述'],axis=1)
        return df


@app.route('/', methods = ['GET', 'POST'])
def index():
    datasets,_,folders = datasetList()
    originalds = []
    for i in range(len(datasets)):
        if folders[i] == 'datasets': originalds += [datasets[i]]
    if request.method == 'POST':
            f = request.files['file']
            f.save(os.path.join('datasets', f.filename))
            return redirect('/')
    return render_template('index.html', originalds = originalds)

@app.route('/datasets/')
def datasets():
    return redirect('/')

@app.route('/datasets/<dataset>')
def dataset(description = None, head = None, dataset = None):
    df = loadDataset(dataset)
    try:
        head = df.head(10)
    except: pass
    return render_template('dataset.html',
                           head = head.to_html(index=False, classes='table table-striped table-hover'),
                           dataset = dataset)

@app.route('/datasets/<dataset>/models')
def models(dataset = dataset):
    clfmodels = algorithms.classificationModels()
    predmodels = algorithms.regressionModels()
    return render_template('models.html', dataset = dataset,
                           clfmodels = clfmodels,
                           predmodels = predmodels)

@app.route('/datasets/<dataset>/modelprocess/', methods=['POST'])
def model_process(dataset = dataset):
    algscore = request.form.get('model')
    alg, score = algscore.split('.')
    data_copy = loadDataset(dataset)
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
    data_analysis = data_copy.drop(['Keyword','职位薪资','公司名称','职位名称'],axis=1)
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
    train_set, test_set = train_test_split(data_analysis, test_size=0.2, random_state=42)
    y_train = np.array(train_set['salary'])
    x_train=train_set.drop(['salary'],axis=1)
    y_test = np.array(test_set['salary'])
    x_test=test_set.drop(['salary'],axis=1)

    if score == 'Classification':
        mod = algorithms.classificationModels()[alg]
        

    elif score == 'Regression':
        mod = algorithms.regressionModels()[alg]
    mod.fit(x_train,y_train)
    y_test_pre = mod.predict(x_test)
    mse = mean_squared_error(y_test, y_test_pre)
    rmse = np.sqrt(mse)
    scores = {}
    scores['test_error'] = str(rmse)
    return render_template('scores.html', scores = scores, dataset = dataset, alg=alg,
                           score = score)
    

@app.route('/datasets/<dataset>/graphs')
def graphs(dataset = dataset):
    data_copy = loadDataset(dataset)
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
    data_analysis = data_copy.drop(['Keyword','职位薪资','公司名称','职位名称'],axis=1)

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

    ds = data_analysis
    columns = ds.columns
    return render_template('graphs.html', dataset = dataset, columns=columns)

@app.route('/datasets/<dataset>/graphprocess/', methods=['POST'])
def graph_process(dataset = dataset):
    histogram = request.form.getlist('histogram')
    pie = request.form.getlist('pie')
    bar = request.form.getlist('bar')
    data_copy = loadDataset(dataset)
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
    data_analysis = data_copy.drop(['Keyword','职位薪资','公司名称','职位名称'],axis=1)

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

    ds = data_analysis
    import plotfunctions as plotfun
    figs = {}
    if histogram != [''] and histogram != []:
        figs['直方图'] = str(plotfun.plot_histsmooth(ds, histogram), 'utf-8')
    if pie != [''] and pie != []:
        figs['饼图'] = str(plotfun.plot_pie(ds, pie), 'utf-8')
    if bar != [''] and bar != []:
        figs['柱状图'] = str(plotfun.plot_bar(ds, bar), 'utf-8')
    if figs == {}: return redirect('/datasets/' + dataset + '/graphs')
    return render_template('drawgraphs.html', figs = figs, dataset = dataset)

@app.route('/datasets/<dataset>/predict')
def predict(dataset = dataset):
    columns = ["公司规模","学历要求","工作经验",'Python', 'Sql', 'R', 'Java', 'Hadoop', 'C', 'Spark', 'Excel',
       'Hive', 'Sas']
    target = ['salary']
    return render_template('predict.html', dataset = dataset,
                           columns = columns,target=target)

@app.route('/datasets/<dataset>/prediction/', methods=['POST'])
def predict_process(dataset = dataset):
    res = request.form.get('response')
    columns = ["公司规模","学历要求","工作经验",'Python', 'Sql', 'R', 'Java', 'Hadoop', 'C', 'Spark', 'Excel',
       'Hive', 'Sas']
    values = {}
    counter = 0
    for col in columns:
        values[col] = request.form.get(col)
        if values[col] != '' and col != res: counter +=1
    
    if counter == 0: return redirect('/datasets/' + dataset + '/predict')
    
    predictors = {}
    for v in values:
        if values[v] != '':
            try: predictors[v] = [float(values[v])]
            except: predictors[v] = [values[v]]
    import shanghaijob as pre
    pre.predict(predictors)
    #return pd.DataFrame(Xpred).to_html()
    predictions = {}
    predictions['Prediction'] = pre.predict(predictors)
    predictors.pop(res, None)
    for p in predictors:
        if str(predictors[p][0]).isdigit() is True: predictors[p] = int(predictors[p][0])
        else:
            try: predictors[p] = round(predictors[p][0],2)
            except: predictors[p] = predictors[p][0]
    for p in predictions:
        if str(predictions[p]).isdigit() is True: predictions[p] = int(predictions[p])
        else:
            try: predictions[p] = round(predictions[p],2)
            except: continue
    if len(predictors) > 15: predictors = {'Number of predictors': len(predictors)}
    #return str(predictors) + res + str(predictions) + alg + score
    
    return render_template('prediction.html', predictions = predictions, response = res,
                           predictors = predictors,
                           dataset = dataset)
@app.route('/datasets/<dataset>/modelargs')
def modelargs(dataset = dataset):
    clfmodels = algorithms.classificationModels()
    predmodels = algorithms.regressionModels()
    return render_template('args.html', dataset = dataset,
                           clfmodels = clfmodels,
                           predmodels = predmodels)

@app.route('/datasets/<dataset>/modelargs1/', methods=['POST'])
def modelarg_process(dataset = dataset):
    algscore = request.form.get('model')
    alg, score = algscore.split('.')
    s1 = request.form.get('args')
    data_copy = loadDataset(dataset)

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
    data_analysis = data_copy.drop(['Keyword','职位薪资','公司名称','职位名称'],axis=1)
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

    importances = xg_reg.feature_importances_
    feature_importances = pd.DataFrame(importances,index = x_train1.columns)
    feature_importances_del = feature_importances[feature_importances[0]==0]

    x_train_tmp = x_train.drop(list(feature_importances_del.index),axis=1)
    x_test_tmp = x_test.drop(list(feature_importances_del.index),axis=1)
    
    columns = data_analysis.columns
    company_features = ['公司规模','区','融资情况']
    employee_features = ['工作经验','学历要求']
    cat_features = company_features+employee_features
    data_analysis_new = data_analysis.copy()
    for x in cat_features:
        data_analysis_tmp = data_analysis.copy()
        rank_df = data_analysis_tmp.loc[:,[x,'salary']].groupby(x).mean()
        rank_df[x] = rank_df.index
        rank_df.columns = [x+'_rank',x]
        data_analysis_new = pd.merge(data_analysis_new,rank_df,on = x,how='left')
    data_analysis_new = pd.get_dummies(data_analysis_new).reset_index(drop=True)
    data_analysis_new = data_analysis_new.drop(['salary']+list(feature_importances_del.index),axis=1)
    x_train_tmp = data_analysis_new.loc[x_train1.index,:]
    x_test_tmp = data_analysis_new.loc[x_test1.index,:]

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


    if score == 'Classification':
        mod = algorithms.classificationModels()[alg]
        

    elif score == 'Regression':
        mod = algorithms.regressionModels()[alg]
    s2 = s1.split(":")
    new={}
    if '.' in s2[1].strip('[]').split(',')[0]:
        new[s2[0]] = list(map(lambda x: float(x),s2[1].strip('[]').split(',')))
    else:
        new[s2[0]] = list(map(lambda x: int(x),s2[1].strip('[]').split(',')))
    param_grid = []
    param_grid.append(new)

    grid_search = GridSearchCV(mod, param_grid, cv=5,
                               scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(x_train_tmp, y_train.ravel())

    cvres = grid_search.cv_results_
    results = []
    for x in zip(cvres["params"],np.sqrt(-cvres["mean_test_score"])):
        results.append(x)
       
    best_params = {}
    best_params['最佳参数'] = grid_search.best_params_

    return render_template('modelargs1.html', scores = results, dataset = dataset, alg=alg,
                           predictors = best_params)
if __name__ == '__main__':
    app.run(debug=False)