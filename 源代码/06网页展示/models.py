# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 20:00:31 2018

@author: Alvaro
"""
def classificationModels():

    models = {}
    
    ######################################
    # Logistic Regression
    ######################################
    from sklearn.linear_model import LogisticRegression
    models['Logistic Regression'] = {}
    models['Logistic Regression'] = LogisticRegression()
    
    ######################################
    # Random Forest
    ######################################
    from sklearn.ensemble import RandomForestClassifier
    models['Random Forests'] = RandomForestClassifier()
    
    ######################################
    # K Nearest Neighbors
    ######################################
    from sklearn.neighbors import KNeighborsClassifier
    models['K Nearest Neighbors'] = KNeighborsClassifier()
    
    ######################################
    # AdaBoost
    ######################################
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    models['Ada Boost'] = AdaBoostClassifier()
    
    ######################################
    # X Gradient Boosting
    ######################################
    from xgboost import XGBClassifier
    models['X Gradient Boosting'] = XGBClassifier()
    
    ######################################
    # Neural Networks: MultiLayer Perceptron
    ######################################
    
    return models


def regressionModels():

    models = {}
    
    ######################################
    # Linear Regression
    ######################################
    from sklearn.linear_model import LinearRegression
    models['Linear Regression'] = {}
    models['Linear Regression'] = LinearRegression()
    
    ######################################
    # Random Forest
    ######################################
    from sklearn.ensemble import RandomForestRegressor
    models['Random Forests'] = RandomForestRegressor()
    
    ######################################
    # K Nearest Neighbors
    ######################################
    from sklearn.neighbors import KNeighborsRegressor
    models['K Nearest Neighbors'] = KNeighborsRegressor()
    
    ######################################
    # AdaBoost
    ######################################
    from sklearn.ensemble import AdaBoostRegressor
    models['AdaBoost'] = AdaBoostRegressor()
    
    ######################################
    # X Gradient Boosting
    ######################################
    from xgboost import XGBRegressor
    models['Xgboost'] = XGBRegressor()
    
    ######################################
    # Neural Networks: MultiLayer Perceptron
    ######################################
    from sklearn.tree import DecisionTreeRegressor   
    models['Decision Tree'] = DecisionTreeRegressor()
    
    from sklearn.ensemble import BaggingRegressor
    models['Bagging'] = BaggingRegressor()
    
    from sklearn.ensemble import GradientBoostingRegressor
    models['Gbdt'] = GradientBoostingRegressor
    
    from sklearn.linear_model import Lasso
    models['Lasso'] = Lasso()
    
    return models