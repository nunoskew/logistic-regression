# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:17:52 2014

@author: sku
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from log_regression import *   


def read_data():
    """() -> DataFrame,DataFrame
    
    Returns panda dataframes train and test from
    train.csv and test.csv
    
    """
    return pd.read_csv('./data/train.csv'),pd.read_csv('./data/test.csv')
    
def one_hot_dataframe(data, cols, replace=False):
    """
    
    Returns DataFrame with categorical features turned into
    binary integer features. (Grabbed this function on stack overflow!)
    
    """
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)
    
def pre_process_(train,test):
    """ (DataFrame,DataFrame) -> DataFrame,DataFrame,matrix,matrix
    
    Does some of the pre-processing steps done in 
    http://trevorstephens.com/post/72916401642/titanic-getting-started-with-r
    Check it out!
    Returns 'train' and 'test' DataFrames, (m,1) matrix 'y_matrix'
    and (m,1) matrix 'id_matrix' for submission purposes!
    
    """
    #first column for submission matrix
    id=test['PassengerId']
    #turn it into a ndarray
    id_matrix=np.array(id).reshape(len(id),1)
    #set up target variable vector
    y=train['Survived']
    #create matrix y_matrix of size (m,1) out of y
    y_matrix=np.array(y).reshape(len(y),1)
    #builds title column
    train,test=build_title(train,test)
    train,test=build_family(train,test)
    #removes columns cabin, passenger_id and name
    train=train.drop(['Cabin'],1)
    test=test.drop(['Cabin'],1)
    train=train.drop(['PassengerId'],1)
    test=test.drop(['PassengerId'],1)
    train=train.drop(['Name'],1)
    test=test.drop(['Name'],1)
    #turn pclass column into a column of strings
    train['Pclass']=train['Pclass'].astype(str)
    test['Pclass']=test['Pclass'].astype(str)
    #remove columns ticket and fare
    train=train.drop(['Ticket'],1)
    test=test.drop(['Ticket'],1)
    train=train.drop(['Fare'],1)
    test=test.drop(['Fare'],1)
    #test['Fare'][152]=test[test['Pclass']==3]['Fare'].median()
    train=train.drop(['Embarked'],1)
    test=test.drop(['Embarked'],1)
    #exclude target variable
    train=train.drop(['Survived'],1)
    #turn categorical features into numerical ones
    train,_,_ =one_hot_dataframe(train,['Pclass','Sex','title'], replace=True)
    test,_,_ =one_hot_dataframe(test,['Pclass','Sex','title'], replace=True)
    #remove redundant column Sex=male
    train=train.drop(['Sex=male'],1)
    test=test.drop(['Sex=male'],1)
    #merge rare title classes into 2
    train['lady']=train['title=Jonkheer']+train['title=Lady']+train['title=the Countess']
    train['boss']=train['title=Col']+train['title=Don']+train['title=Major']
    test['lady']=test['title=Dona']+test['title=Ms']
    test['boss']=test['title=Col']
    #drop rare classes
    train=train.drop(['title=Capt'],1)
    train=train.drop(['title=Col'],1)
    train=train.drop(['title=Don'],1)
    train=train.drop(['title=Jonkheer'],1)
    train=train.drop(['title=Lady'],1)
    train=train.drop(['title=Major'],1)
    train=train.drop(['title=Mlle'],1)
    train=train.drop(['title=Mme'],1)
    train=train.drop(['title=Ms'],1)
    train=train.drop(['title=Sir'],1)
    train=train.drop(['title=the Countess'],1)    
    test=test.drop(['title=Dona'],1)
    test=test.drop(['title=Col'],1)
    test=test.drop(['title=Ms'],1)

    return train,test,y_matrix,id_matrix
    
def build_family(train,test):
    """ (DataFrame,DataFrame) -> DataFrame,DataFrame
    
    Merges Sibsp and Parch features into a Family feature 
    which is the sum of siblings, spouse, parents, children of a person. 
    Returns 'train' and 'test' DataFrames.
    
    """
    #merge sibsp and parch features into a family feature 
    # sum of siblings, spouse, parents, children
    train['Family']=train['SibSp']+train['Parch']
    test['Family']=test['SibSp']+test['Parch']
    #remove columns sibsp and parch
    train=train.drop(['SibSp'],1)
    train=train.drop(['Parch'],1)
    test=test.drop(['SibSp'],1)
    test=test.drop(['Parch'],1)
    
    return train,test
    
def build_title(train,test):
    """ (DataFrame,DataFrame) -> DataFrame,DataFrame
    
    Builds 'title' feature from processing the 'name' feature.
    First word before ','.
    Returns 'train' and 'test' DataFrames.
    
    """
    
    train['title']=[str.split(str.split(name,',')[1],'.')[0].strip() for name in train['Name']]
    test['title']=[str.split(str.split(name,',')[1],'.')[0].strip() for name in test['Name']]
    
    return train,test
    
def turn_into_matrices(train,test):
    """ (DataFrame,DataFrame) -> matrix,matrix
    
    Converts train and test DataFrames into matrices.
    
    """
    return train.as_matrix(),test.as_matrix()