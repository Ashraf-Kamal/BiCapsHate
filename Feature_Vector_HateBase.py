# -*- coding: utf-8 -*-
#!/usr/bin/python
from __future__ import absolute_import
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import csv
import nltk
import os
from textblob import TextBlob
import re
import sys
import textstat
import math
import liwc
from collections import Counter

sentences=[]
Final_Vector=[]
                
def Hate_Score(row,k):
        try:
                df = pd.read_csv(r'HateBase\HateBase_Merged_Final.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['is_Hateful'].loc[df['term'] == word])                       
                        for line in list(df1):
                                print (line)
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Hate Score function')

def Unambigious_Score(row,k):
        try:
                df = pd.read_csv(r'HateBase\HateBase_Merged_Final.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['is_unambiguous'].loc[df['term'] == word])                       
                        for line in list(df1):
                                print (line)
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Unambigious Score function')

def Offensiveness_Score(row,k):
        try:
                df = pd.read_csv(r'HateBase\HateBase_Merged_Final.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['Normalize_Avg_offensiveness'].loc[df['term'] == word])                       
                        for line in list(df1):
                                print (line)
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Offensiveness Score function')

def Nationality_score(row,k):
        try:
                df = pd.read_csv(r'HateBase\HateBase_Merged_Final.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['is_about_nationality'].loc[df['term'] == word])                       
                        for line in list(df1):
                                print (line)
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Nationality Score function')

def Ethnicity_score(row,k):
        try:
                df = pd.read_csv(r'HateBase\HateBase_Merged_Final.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['is_about_ethnicity'].loc[df['term'] == word])                       
                        for line in list(df1):
                                print (line)
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Ethnicity Score function')

def Religion_score(row,k):
        try:
                df = pd.read_csv(r'HateBase\HateBase_Merged_Final.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['is_about_religion'].loc[df['term'] == word])                       
                        for line in list(df1):
                                print (line)
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Religion Score function')

def Gender_score(row,k):
        try:
                df = pd.read_csv(r'HateBase\HateBase_Merged_Final.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['is_about_gender'].loc[df['term'] == word])                       
                        for line in list(df1):
                                print (line)
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Gender Score function')                

def Sexual_Orientation_score(row,k):
        try:
                df = pd.read_csv(r'HateBase\HateBase_Merged_Final.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['is_about_sexual_orientation'].loc[df['term'] == word])                       
                        for line in list(df1):
                                print (line)
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Sexual Orientation Score function')
                
def Disability_score(row,k):
        try:
                df = pd.read_csv(r'HateBase\HateBase_Merged_Final.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['is_about_disability'].loc[df['term'] == word])                       
                        for line in list(df1):
                                print (line)
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Disability Score function')

def Class_score(row,k):
        try:
                df = pd.read_csv(r'HateBase\HateBase_Merged_Final.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['is_about_class'].loc[df['term'] == word])                       
                        for line in list(df1):
                                print (line)
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Class Score function')
        
#read_file = pd.read_excel (r'..\Linguistic_Features\Hatebase\HateBase_Merged_Final.xlsx')
#read_file.to_csv (r'..\Linguistic_Features\Hatebase\HateBase_Merged_Final.csv', index = None, header=True)

#with open(r'..\Final_Datasets\Davidson_17\Davidson_Balanced.csv', 'r') as my_file:
#with open(r'..\Final_Datasets\Founta_18\Founta_Balanced.csv', 'r') as my_file:
#with open(r'..\Final_Datasets\Gao_17\Gao_Balanced.csv', 'r') as my_file:
#with open(r'..\Final_Datasets\Roy_20\Roy_Balanced.csv', 'r') as my_file:
#with open(r'..\Final_Datasets\HateXplain\HateXplain_Bi_Balance.csv', 'r') as my_file:                
#with open(r'..\Final_Datasets\Davidson_17\Davidson_Unbalanced.csv', 'r') as my_file:
#with open(r'..\Final_Datasets\Founta_18\Founta_Unbalanced.csv', 'r') as my_file:
#with open(r'..\Final_Datasets\Gao_17\Gao_Unbalanced.csv', 'r') as my_file:
with open(r'..\Final_Datasets\Roy_20\Roy_Unbalanced.csv', 'r') as my_file:
#with open(r'..\Final_Datasets\HateXplain\HateXplain_Bi_Imbalance.csv', 'r') as my_file:
#with open(r'..\Final_Datasets\HateXplain\HateXplain_Multi_Preprocess.csv', 'r') as my_file:
#with open(r'..\Final_Datasets\Davidson_17\Davidson_Multi_Preprocess.csv', 'r') as my_file:
#with open(r'..\Final_Datasets\Founta_18\Founta_Multi_Preprocess.csv', 'r') as my_file:        
        
    f = csv.reader(my_file, quoting=csv.QUOTE_ALL)
    reader = next(f) 
    Tag_list = list(f)
    #print (Tag_list)
for sublst in Tag_list:
	Final_Vector.append([])        
k=0
for sublst in Tag_list:
  #print (sublst)
  label = sublst[1]
  temp=' '.join(map(str, sublst))  
  Hate_Score(temp,k)
  Unambigious_Score(temp,k)
  Offensiveness_Score(temp,k)
  Nationality_score(temp,k)
  Ethnicity_score(temp,k)
  Religion_score(temp,k)
  Gender_score(temp,k)
  Sexual_Orientation_score(temp,k)
  Disability_score(temp,k)
  Class_score(temp,k)
  Final_Vector[k].append(label)
  #print (k)
  k=k+1
#print (Final_Vector)

#with open(r'FV_Results\Davidson_Balanced1.csv', 'w', newline='') as f:        
#with open(r'FV_Results\Founta_Balanced1.csv', 'w', newline='') as f:
#with open(r'FV_Results\Gao_Balanced1.csv', 'w', newline='') as f:
#with open(r'FV_Results\Roy_Balanced1.csv', 'w', newline='') as f:
#with open(r'FV_Results\HateXplain_Balanced1.csv', 'w', newline='') as f:
#with open(r'FV_Results\Davidson_Unbalanced1.csv', 'w', newline='') as f:
#with open(r'FV_Results\Founta_Unbalanced1.csv', 'w', newline='') as f:
#with open(r'FV_Results\Gao_Unbalanced1.csv', 'w', newline='') as f:
with open(r'FV_Results\Roy_Unbalanced1.csv', 'w', newline='') as f:
#with open(r'FV_Results\HateXplain_Unbalanced1.csv', 'w', newline='') as f:
#with open(r'FV_Results\HateXplain_Multi_Preprocess1.csv', 'w', newline='') as f:
#with open(r'FV_Results\Davidson_Multi_Preprocess1.csv', 'w', newline='') as f:
#with open(r'FV_Results\Founta_Multi_Preprocess1.csv', 'w', newline='') as f:         
        writer = csv.writer(f)
        writer.writerows(Final_Vector)
print (Final_Vector)

print ('Success')


