# -*- coding: utf-8 -*-
"""
Created on Tue 

@author: 
"""

import pandas as pd
from pandas import Series
data=pd.read_csv("job_total.csv",encoding='gbk')
d_jobs=[]
with open("无关岗位.txt") as f:
    for line in f:
        line_split = line.strip()        
        d_jobs.append(line_split)

criterion = lambda row: row['job'] not in d_jobs
data2= data[data.apply(criterion, axis=1)]

data2.columns
data2=data2.drop(['signal '],axis=1)
data2 = data2.drop_duplicates()

import numpy as np
nulls = np.sum(data2.isnull())
nullcols = nulls.loc[(nulls != 0)]
print(nullcols)

finaldata=data2.copy().reset_index()

import re
# 均值函数
def average(job_salary):
    # 取薪资均值----------------
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
    # 函数返回值
    return avg_salary


average(finaldata['salary'].head()[0])



salary_list = []
for i in range(0,len(finaldata)):
    avg_sal = average(finaldata['salary'][i])
    salary_list.append(avg_sal)

sal = Series(salary_list)
sal.head()

finaldata.insert(8,'salarynew',sal)
finaldata.head()
finaldata=finaldata.drop(["index"],axis=1)



job_data = finaldata[['province','city','job','salarynew','company','scale','education','experience']]
job_data.head()
job_data.shape 
jobs = job_data.drop_duplicates()
jobs.head()
jobs.shape
jobs.info()
jobs['scale'].value_counts()
jobs['company'].describe()
#jobs['salarynew'].value_counts().head()
jobs['education'].value_counts()
jobs['salarynew'].describe()
jobs.columns = ['province','city','job','salary','company','scale','education','experience']

%matplotlib inline
import matplotlib.pyplot as plt
jobs.hist(bins=20, figsize=(8,5))
# save_fig("attribute_histogram_plots")
plt.show()





%matplotlib inline
import matplotlib.pyplot as plt
jobs.hist(bins=50, figsize=(8,5))
# save_fig("attribute_histogram_plots")
plt.show()

jobs['job'].value_counts().head()
jobs.info()

grouped_province = jobs['salary'].groupby(jobs['province'])
grouped_province
grouped_province.mean().sort_values(ascending=False).round(2).head()
grouped_province.mean()

from pyecharts import Bar
x_prov = grouped_province.mean().sort_values(ascending=False).index
y_prov = grouped_province.mean().sort_values(ascending=False).round(1).values
bar = Bar("",width=1000, height=600)
bar.add("各省数据分析师月均薪资", x_prov, y_prov,mark_line=["average"],is_label_show=True,is_more_utils=True
        ,xaxis_interval=0, xaxis_rotate=30, yaxis_min=4,yaxis_rotate=30)
bar.render("薪资柱状图.html")






from pyecharts import Map

map = Map("全国数据分析师薪资", width=1200, height=600)
map.add("", x_prov, y_prov,visual_range=[min(y_prov), max(y_prov)],is_piecewise=True, visual_text_color="#fff",symbol_size=15, is_visualmap=True,is_label_show=True)

map.render()


grouped_city = jobs['salary'].groupby(jobs['city'])
grouped_city.mean().sort_values(ascending=False).round(2).head()
jobs_gd = jobs[jobs['province']=='广东']
grouped_gd = jobs_gd['salary'].groupby(jobs_gd['city'])
grouped_gd.mean().sort_values(ascending=False).round(2).head()
x_gd = grouped_gd.mean().sort_values(ascending=False).index
y_gd = grouped_gd.mean().sort_values(ascending=False).values.round(2)
from pyecharts import Map
x_gd=[x+'市' for x in x_gd]
map = Map("广东数据分析师薪资", width=1200, height=600)
map.add("", x_gd, y_gd,maptype="广东",visual_range=[10,19], is_visualmap=True, visual_text_color="#000",
    is_map_symbol_show=False,is_piecewise=True,is_label_show=True)

map.render("广东.html")


grouped_degree = jobs['salary'].groupby(jobs['education'])
grouped_degree.mean().sort_values(ascending=False).round(2).head()

x_degree = grouped_degree.mean().sort_values(ascending=False).index
y_degree = grouped_degree.mean().sort_values(ascending=False).round(1).values
bar = Bar("",width=1000, height=600)
bar.add("月均薪资", x_degree, y_degree,mark_line=["average"],is_label_show=True,is_more_utils=True
        ,xaxis_interval=0, xaxis_rotate=30, yaxis_min=0,yaxis_rotate=30)

bar.render("不同学历数据分析师薪资.html")

import matplotlib
myfont = matplotlib.font_manager.FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
new=grouped_degree.size()
counts = [x/np.sum(new.values)+0.2 for x in new.values] 
cax = jobs.boxplot(column='salary', by='education', widths=counts) 
cax.set_xticklabels(['%s'%k for k in new.index],fontproperties = myfont) 



grouped_scale = jobs['salary'].groupby(jobs['scale'])
grouped_scale.mean().sort_values(ascending=False).round(2).head()
x_scale = grouped_scale.mean().sort_values(ascending=False).index
y_scale = grouped_scale.mean().sort_values(ascending=False).round(1).values
bar = Bar("",width=700, height=500)
bar.add("月均薪资", x_scale, y_scale,mark_line=["average"],is_label_show=True,is_more_utils=True
        ,xaxis_interval=0, xaxis_rotate=30, yaxis_min=5,yaxis_rotate=30)
bar.render("不同规模企业中数据分析师薪资.html")













