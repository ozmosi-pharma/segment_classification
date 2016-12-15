# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:21:13 2016

@author: Sonam Gupta
"""

import pandas as pd
#import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import sys
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
import nltk.classify.util
from collections import OrderedDict


parser = argparse.ArgumentParser()

parser.add_argument('-classifierType', action='store', type=str, default="naiveBayes", dest="classifierType", help="Select which classifier to use ['naiveBayes','randomForest','knn5']. Default(CLASSIFIERTYPE=naiveBayes)")

args = parser.parse_args()
print ("Args: ", str(args))

# To choose from whichever classifier you want to train the model on

if args.classifierType not in ["naiveBayes", "randomForest", "knn5"]:
    print ("ERROR: Please pick an appropriate classification algorithm from the set supported.")
    print (["naiveBayes", "randomForest", "knn5"])
    quit()

def getTrainedCLassifier(classifierType, train):
    if classifierType == "naiveBayes":
        from nltk.classify import NaiveBayesClassifier
        trainedClassifier = NaiveBayesClassifier.train(train)
    elif classifierType == "randomForest":
        from sklearn.ensemble import RandomForestClassifier as rfc
        trainedClassifier = SklearnClassifier(rfc(n_estimators=25, n_jobs = 2))
        trainedClassifier.train(train)
    elif classifierType == "knn5":
        from sklearn.neighbors import KNeighborsClassifier as knn
        trainedClassifier = SklearnClassifier(knn(5))
        trainedClassifier.train(train)
    return trainedClassifier

'''
segment_df = pd.read_csv("Result.csv", encoding = 'cp1252', error_bad_lines = False)
# replace na's with space
segment_df.fillna('',inplace = True)'''

segment_classify = open("Result.csv", encoding = 'cp1252')
result_labels = []
# Final class labels for each row
final_label = []
#keyword identifiers
patient_segment = ['Stage I', 'Stage II', 'Stage III', 'Stage IV', 'metastatic', 'advanced', 'Neoadjuvant', 'Adjuvant', 'first line']
patient_population = ['surgery', 'therapy', 'brachytherapy', 'radiotherapy', 'no prior tx', 'no metastases', 'prior endocrine', 'androgen independent', 'bcr', 'failed local tx', 'mets', 'hormone refractory', 'androgen independent']
inclusion_criteria = ['status of prostatectomy', 'negative nodes', 'metastatic', 'locally advanced', 'failed local tx', 'no tumor']


headers = segment_classify.readline().split(',')

for line in segment_classify:
    line = line.split(',')
    result_label = line[5]
    # features that will help classify, are the columns in csv file
    feature_col_segment = line[2]    
    feature_col_inclusion = line[3]
    feature_col_population = line[4]
    
    if (result_label != "NA") and (result_label != "") and (feature_col_segment != "\r\n") and (feature_col_segment != "(N/A)") and (feature_col_segment != "\n") and (feature_col_inclusion != "\r\n") and (feature_col_inclusion != "N/A") and (feature_col_inclusion != "\n") and (feature_col_population != "\r\n") and (feature_col_population != "N/A") and (feature_col_population != "\n"):
       if result_label == 0 and feature_col_segment in patient_segment and feature_col_inclusion in inclusion_criteria and feature_col_population in patient_population:
           
           result_labels.append()
       
    

'''
names_classifiers = ["Random Forest", "Naive Bayes", "Nearest Neighbors"]

classifiers = [KNeighborsClassifier(3),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               GaussianNB()]

# training and testing split of the dataset               
for count, ds in enumerate(segment_df):
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size = .2, random_state = 42)
'''
