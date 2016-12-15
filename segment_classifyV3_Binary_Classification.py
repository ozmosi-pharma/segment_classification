# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:25:15 2016

@author: Sonam Gupta
"""

import pandas as pd

# Reading in the file v2
cancer_df = pd.read_csv("Result.csv", encoding = 'cp1252', error_bad_lines = False)
cancer_df.fillna('',inplace = True)


# labels and identifiers

#labels = ["No", "priorlocaltx"]
labels = [2, 3]
key_segment = ['Stage I', 'Stage II', 'Stage III', 'Stage IV', 'metastatic', 'advanced', 'Neoadjuvant', 'Adjuvant', 'first line']
key_inclusion = ['status of prostatectomy', 'negative nodes', 'metastatic', 'locally advanced', 'failed local tx', 'no tumor']
key_population = ['surgery', 'therapy', 'brachytherapy', 'radiotherapy', 'no prior tx', 'no metastases', 'prior endocrine', 'androgen independent', 'bcr', 'failed local tx', 'mets', 'hormone refractory', 'androgen independent']


cancer_df.insert(6, 'No prior category', 0, allow_duplicates = False)
#print(cancer_df.head())

for i in range(1, len(cancer_df)):
    if cancer_df.at[5, 'Result'] == '0':
        for word in key_segment, key_inclusion, key_population:
            if word in (cancer_df.iloc[i]['Patient Segment(s)'], cancer_df.iloc[i]['Inclusion Criteria'], cancer_df.iloc[i]['Patient Population']):
                cancer_df.at[i,'No prior category'] = labels[0]
            #print(cancer_df['Patient Segment(s)'].dtype)
    
        for word in key_segment, key_inclusion, key_population:
            if word in (cancer_df.iloc[i]['Patient Segment(s)'], cancer_df.iloc[i]['Inclusion Criteria'], cancer_df.iloc[i]['Patient Population']):
                cancer_df.at[i,'No prior category'] = labels[1]
        
    cancer_df.to_csv('Result_localized_classification.csv')
