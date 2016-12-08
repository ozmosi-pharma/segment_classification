# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:03:05 2016

@author: Sonam Gupta
"""
import pandas as pd

# Reading in the file v2
cancer_df = pd.read_csv("disease_segment.csv", encoding = 'cp1252', error_bad_lines = False)
cancer_df.fillna('',inplace = True)


# labels and identifiers

#labels = ['No prior local tx', 'prior local tx']
labels = [0, 1]
keyword_identifiers_no = ['surgery', 'radiotherapy', 'brachytherapy', 'Newadjuvant', 'neo', 'Adjuvant', 'Mets at first diagnosis', 'M1HSPC']
keyword_identifiers_yes = ['no prior tx', 'BCR', 'Neoadjuvant', 'neo', 'Adjuvant', 'Hormone refractory', 'Mets', 'No mets']

cancer_df.insert(2, 'Result', 0, allow_duplicates = False)
#print(cancer_df.head())
for i in range(1, len(cancer_df)):

    
    for word in keyword_identifiers_no:
        if word in cancer_df.iloc[i]['Patient Segment(s)']:
            
            cancer_df.at[i,'Result'] = labels[0]
            #print(cancer_df['Patient Segment(s)'].dtype)

    for word in keyword_identifiers_yes:
        if word in cancer_df.iloc[i]['Patient Segment(s)']:
            cancer_df.at[i,'Result']= labels[1]
    
    cancer_df.to_csv('Result.csv')