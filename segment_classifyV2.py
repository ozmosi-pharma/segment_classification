# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:34:39 2016

@author: Sonam Gupta
"""

import numpy as np
import csv
from sklearn.naive_bayes import GaussianNB

'''
patient_segment = ['Stage I', 'Stage II', 'Stage III', 'Stage IV', 'metastatic', 'advanced', 'Neoadjuvant', 'Adjuvant', 'first line']
patient_population = ['surgery', 'therapy', 'brachytherapy', 'radiotherapy', 'no prior tx', 'no metastases', 'prior endocrine', 'androgen independent', 'bcr', 'failed local tx', 'mets', 'hormone refractory', 'androgen independent']
inclusion_criteria = ['status of prostatectomy', 'negative nodes', 'metastatic', 'locally advanced', 'failed local tx', 'no tumor']
'''

with open(r"Result.csv") as segment_classify:
    reader = csv.reader(segment_classify, delimiter=',', quotechar = '"')
    headers = segment_classify.readline().split(',')
    
    for row in reader:
        
        if row:
            result_label = row[5]
            # features that will help classify, are the columns in csv file
            feature_col_segment = row[2]    
            feature_col_inclusion = row[3]
            feature_col_population = row[4]
        
            np.column_stack((feature_col_segment, feature_col_inclusion, feature_col_population, result_label))
            a = np.array(feature_col_segment)
            b = np.array(feature_col_inclusion)
            c = np.array(feature_col_population)
            d = np.array(result_label)
            
            x = [a,b,c,d]
            y = ['localized', 'm1hspc', 'bcr_m0', 'mcrpc']
            
    model = GaussianNB()
    
    model.fit(x, y)
    
    print (model)
    
    #predicted = model.predict(x)
    
            