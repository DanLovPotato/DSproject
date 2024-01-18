# -*- coding: utf-8 -*-
"""
Created on Sat 

@author: 
"""

import requests
import re
from lxml import etree
import time
import random
import pandas as pd
from pandas import DataFrame
# -------------
city_data = pd.read_csv('city.csv')
data = city_data.loc[42:48][['code', '省', '市']]
# print(data.values)
citylist = list(data['市'])


# 页面获取函数

def get_page(page, city_code):
    header = {
        'User-Agent': 'Mozilla/5.0(WindowsNT6.1;rv:2.0.1)Gecko/20100101Firefox/4.0.1'
    }
   
    print('正在爬取第', page, '页')
    url = 'https://www.zhipin.com/c{code}-p100511/?page={page}&ka=page-{page}'.format(code=city_code, page=page)
    response = requests.get(url, headers=header)
    return response.text
# --------------


# 页面解析函数
def parse(html, city, provence, page):
    # data = json.loads(data)
    # print(data)
    # 观察数据结构可得
    data = etree.HTML(html)
    # 取工资均值
    items = data.xpath('//*[@id="main"]/div/div[2]/ul/li')
    for item in items:
        job_title = item.xpath('./div/div[1]/h3/a/div[1]/text()')[0]
        job_salary = item.xpath('./div/div[1]/h3/a/span/text()')[0]
        job_company = item.xpath('./div/div[2]/div/h3/a/text()')[0]
        job_experience = item.xpath('./div/div[1]/p/text()[2]')[0]
        job_degree = item.xpath('./div/div[1]/p/text()[3]')[0]
        company_scale = item.xpath('./div/div[2]/div/p/text()[3]')
        # 取薪资均值----------------
        avg_salary = average(job_salary)
        # -------------------------
        signal = city + str(page)
        print(province, '|', city, '|', job_title, '|', job_salary, '|', job_company, '|', job_experience, '|', job_degree, '|', company_scale,
              '|', avg_salary)
        job = {
            'signal': signal,
            '省': province,
            '城市': city,
            '职位名称': job_title,
            '职位薪资': job_salary,
            '公司名称': job_company,
            '工作经验': job_experience,
            '学历要求': job_degree,
            '公司规模': company_scale
        }
        df=DataFrame(job)
        df.to_csv('F:/my_csv.csv', mode='a', header=False,index=None)
# ---------------------------------------


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


# 连接到MongoDB



# -----------------




def jobspider(city_code, city, provence):
    # 最大爬取页数
    MAX_PAGE = 8
    for i in range(1, MAX_PAGE + 1):
        try:
            html = get_page(i, city_code)
            # ------------ 解析数据 ---------------
            parse(html, city, provence, i)
            print('-' * 100)
            time.sleep(random.randint(0, 10))
        except Exception:
            break



for city in citylist:
    city_code =data[data['市']==city]['code'].values[0]
    province = data[data['市']==city]['省'].values[0]
    # 职位爬虫
    jobspider(city_code, city, province)
        # -----------------