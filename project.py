#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:46:19 2017

@author: Joel_Quek
"""


import numpy
import pandas
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# load dataset
dataframe = pandas.read_csv("train.csv")


dataframe.job.replace(('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)

dataframe.marital.replace(('divorced','married','single','unknown'),(1,2,3,4), inplace=True)
dataframe.education.replace(('basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown'),(1,2,3,4,5,6,7,8), inplace=True)
dataframe.default.replace(('no','yes','unknown'),(1,2,3), inplace=True)
dataframe.housing.replace(('no','yes','unknown'),(1,2,3), inplace=True)
dataframe.loan.replace(('no','yes','unknown'),(1,2,3), inplace=True)
dataframe.contact.replace(('cellular','telephone'),(1,2), inplace=True)
dataframe.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
dataframe.day_of_week.replace(('mon','tue','wed','thu','fri'),(1,2,3,4,5), inplace=True)
dataframe.y.replace(('yes','no'),(0,1), inplace=True)
dataframe.poutcome.replace(('failure','success', 'nonexistent'),(0,1,2), inplace=True)

dataframe.to_csv('out.csv')


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier

from sklearn.cross_validation import train_test_split
X, y = dataframe.iloc[:,:-1], dataframe.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)




from sklearn.metrics import matthews_corrcoef

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
clf4 = XGBClassifier()
clf5 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=seed)
clf6 = BaggingClassifier(base_estimator=clf1, n_estimators=100, random_state=seed)
clf7 = BaggingClassifier(base_estimator=clf2, n_estimators=100, random_state=seed)
clf8 = BaggingClassifier(base_estimator=clf3, n_estimators=100, random_state=seed)

clf9 = BaggingClassifier(base_estimator=clf4, n_estimators=100, random_state=seed)


#eclf = VotingClassifier(estimators=[('lr', clf6), ('rf', clf7), ('gnb', clf8),('xgb', clf9),('bagDecisionTree', clf5)], voting='hard')
#tested previous but not better
eclf = VotingClassifier(estimators=[('rf', clf2), ('xglb', clf4)], voting='hard')


eclf.fit(X_train, y_train)

Y_Pred = eclf.predict(X_test)
matthews_corrcoef(y_test, Y_Pred)





