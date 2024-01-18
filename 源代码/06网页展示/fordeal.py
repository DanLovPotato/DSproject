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
    param_grid = request.form.get(col)
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
    param_grid = [   
    {'alpha':[0.05,0.06,0.1,0.11,0.12,0.13,0.15,0.17,0.2,0.4] }
      ]

    grid_search = GridSearchCV(mod, param_grid, cv=5,
                               scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(x_train_tmp, y_train.ravel())

    cvres = grid_search.cv_results_
    results = {}
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        results[params] = np.sqrt(-mean_score)
       
    best_params = {}
    best_params['最佳参数'] = grid_search.best_params_

    return render_template('参数结果.html', scores = results, dataset = dataset, alg=alg,
                           score = best_params)