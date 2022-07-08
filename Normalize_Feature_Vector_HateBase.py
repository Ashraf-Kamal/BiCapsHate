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
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#df = pd.read_csv('FV_Results\Davidson_Balanced.csv')
#df = pd.read_csv('FV_Results\Founta_Balanced1.csv')
#df = pd.read_csv('FV_Results\Gao_Balanced1.csv')
#df = pd.read_csv('FV_Results\Roy_Balanced1.csv')
#df = pd.read_csv('FV_Results\HateXplain_Balanced1.csv')

#df = pd.read_csv('FV_Results\Davidson_Unbalanced1.csv')
df = pd.read_csv('FV_Results\Founta_Unbalanced.csv')
#df = pd.read_csv('FV_Results\Gao_Unbalanced1.csv')
#df = pd.read_csv('FV_Results\Roy_Unbalanced1.csv')
#df = pd.read_csv('FV_Results\HateXplain_Unbalanced1.csv')
#df = pd.read_csv('FV_Results\Davidson_Multi_Preprocess1.csv')
#df = pd.read_csv('FV_Results\Founta_Multi_Preprocess.csv')


#make a copy of dataframe
scaled_features = df.copy()

#col_names = ['F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23', 'F24', 'F25', 'F26']
col_names = ['F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16']
features = scaled_features[col_names]

# Use scaler of choice; here Standard scaler is used
scaler = MinMaxScaler().fit(features.values)
features = scaler.transform(features.values)

scaled_features[col_names] = features
print (scaled_features)
print (type(scaled_features))
#scaled_features.to_csv(r'FV_Results\Davidson_Balanced111.csv',index=False, header=False)
#scaled_features.to_csv(r'FV_Results\Founta_Balanced11.csv',index=False, header=False)
#scaled_features.to_csv(r'FV_Results\Gao_Balanced11.csv',index=False, header=False)
#scaled_features.to_csv(r'FV_Results\Roy_Balanced11.csv',index=False, header=False)
#scaled_features.to_csv(r'FV_Results\HateXplain_Balanced11.csv',index=False, header=False)

#scaled_features.to_csv(r'FV_Results\Davidson_Unbalanced11.csv',index=False, header=False)
scaled_features.to_csv(r'FV_Results\Founta_Unbalanced111.csv',index=False, header=False)
#scaled_features.to_csv(r'FV_Results\Gao_Unbalanced11.csv',index=False, header=False)
#scaled_features.to_csv(r'FV_Results\Roy_Unbalanced11.csv',index=False, header=False)
#scaled_features.to_csv(r'FV_Results\Davidson_Multi_Preprocess1.csv',index=False, header=False)
#scaled_features.to_csv(r'FV_Results\Founta_Multi_Preprocess111.csv',index=False, header=False)

print ('Success')
