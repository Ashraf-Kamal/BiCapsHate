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

#This function is used to calcualte the polarity score of a text.
def Polarity_Score(text,k):
    try:
        #print (text,k)
        pol_score = TextBlob(text)
        val="{:.3f}".format(pol_score.sentiment.polarity)
        #print ('Polarity score:', pol_score.sentiment.polarity)
        Final_Vector[k].append(val)
    except:
        print('Exception occurs in Polarity_Score function')

#This function is used to calcualte the subjectivity score of a text.
def Subjectivity_Score(text,k):
    try:
        subjectivity_score = TextBlob(text)
        val="{:.3f}".format(subjectivity_score.sentiment.subjectivity)
        #print ('Subjectivity score:', subjectivity_score.sentiment.subjectivity)
        Final_Vector[k].append(val)
    except:
        print('Exception occurs in Subjectivity_Score function')

#This function is used to calcualte the number of positive words in a text.
def Positive_Words_Count(text,k):
    try:
        pos_word_list=[]
        token = word_tokenize(text)
        for word in token:               
            testimonial = TextBlob(word)
            if testimonial.sentiment.polarity >= 0.5:
                pos_word_list.append(word)
                
        Final_Vector[k].append(len(pos_word_list))            
    except:
        print("Exception occur in Positive_Words_Count function")

#This function is used to calcualte the number of negative words in a text.
def Negative_Words_Count(text,k):
    try:
        neg_word_list=[]
        token = word_tokenize(text)    
        #print (token)
        for word in token:               
            testimonial = TextBlob(word)
            if testimonial.sentiment.polarity <= -0.5:
                neg_word_list.append(word)
            
        Final_Vector[k].append(len(neg_word_list))            
    except:
        print("Exception occur in Negative_Words_Count function")


#This function is used to calcualte the number of interjections in a text.
def Interjection_count(row,k):
        try:
                count=0
                text = word_tokenize(row)                
                pos=nltk.pos_tag(text)
                selective_pos = ['UH']
                #for word,tag in pos:
                        #print (tag)
                selective_pos_words = []
                for word,tag in pos:
                        if tag in selective_pos:
                                selective_pos_words.append((word,tag))
                                count+=1
                if (count>2):
                    Final_Vector[k].append(1)
                else:
                    Final_Vector[k].append(0)                                                        
        except:
                print("Exception occur in Interjection_Count function")

#This function is used to calcualte the number of noun (singular or mass) in a text.
def Noun_count(row,k):
        try:
                count=0
                text = word_tokenize(row)                
                pos=nltk.pos_tag(text)
                selective_pos = ['NN']
                #for word,tag in pos:
                        #print (tag)
                selective_pos_words = []
                for word,tag in pos:
                        if tag in selective_pos:
                                selective_pos_words.append((word,tag))
                                count+=1
                if (count>2):
                    Final_Vector[k].append(1)
                else:
                    Final_Vector[k].append(0)                                                        
        except:
                print("Exception occur in Interjection_Count function")

#This function is used to calcualte the number of verb (base form) in a text.
def Verb_count(row,k):
        try:
                count=0
                text = word_tokenize(row)                
                pos=nltk.pos_tag(text)
                selective_pos = ['VB']
                #for word,tag in pos:
                        #print (tag)
                selective_pos_words = []
                for word,tag in pos:
                        if tag in selective_pos:
                                selective_pos_words.append((word,tag))
                                count+=1
                if (count>2):
                    Final_Vector[k].append(1)
                else:
                    Final_Vector[k].append(0)                                                        
        except:
                print("Exception occur in Interjection_Count function")


#This function is used to calcualte the number of adverbs in a text.
def Adverb_count(row,k):
        try:
                count=0
                text = word_tokenize(row)                
                pos=nltk.pos_tag(text)
                selective_pos = ['RB']
                #for word,tag in pos:
                        #print (tag)
                selective_pos_words = []
                for word,tag in pos:
                        if tag in selective_pos:
                                selective_pos_words.append((word,tag))
                                count+=1
                if (count>2):
                    Final_Vector[k].append(1)
                else:
                    Final_Vector[k].append(0)                            
        except:
                print("Exception occur in Adverb_Count function")   
                
#This function is used to calcualte the number of adjectives in a text.
def Adjective_count(row,k):
        try:
                count=0
                text = word_tokenize(row)                
                pos=nltk.pos_tag(text)
                selective_pos = ['JJ']
                #for word,tag in pos:
                #       print (tag)
                selective_pos_words = []
                for word,tag in pos:
                        if tag in selective_pos:
                                selective_pos_words.append((word,tag))
                                count+=1                        
                if (count>2):
                    Final_Vector[k].append(1)
                else:
                    Final_Vector[k].append(0)                                                        
        except:
                print("Exception occur in Adjective_count function")   

#This function is used to calcualte the sum of affective valence scores of all tokens in a text.  
def Affective_Valence_Score(row,k):
        try:
                df = pd.read_csv(r'Affective\all.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['Valence Mean'].loc[df['Description'] == word])                       
                        for line in list(df1):
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Affective_Valence_Score function')

#This function is used to calcualte the sum of affective arousal scores of all tokens in a text.  
def Affective_Arousal_Score(row,k):
        try:
                df = pd.read_csv(r'Affective\all.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['Arousal Mean'].loc[df['Description'] == word])                       
                        for line in list(df1):
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Affective_Arousal_Score function')
                
#This function is used to calcualte the sum of affective dominance scores of all tokens in a text.  
def Affective_Dominance_Score(row,k):
        try:
                df = pd.read_csv(r'Affective\all.csv',delimiter=',',encoding='latin-1')        
                res=0
                token = word_tokenize(row)                
                for word in token:                
                        df1=(df['Dominance Mean'].loc[df['Description'] == word])                       
                        for line in list(df1):
                                res+=line                                
                #res=res/len(token)
                Final_Vector[k].append(res)                            
        except:
                print('Exception occurs in Affective_Dominance_Score function')

#This function is used to calcualte the POS_Stylistics features of a text.        
def POS_Stylistics(row,k):
        try:
                count_noun=0
                count_pronoun=0
                count_verb=0
                count_adjective=0
                count_adverb=0
                count_determiner=0
                count_conjuction=0
                count_cardinal=0
                count_existential=0
                count_foreign =0
                count_listitem =0
                count_modal =0
                count_possessive =0
                count_particle =0
                count_symbol =0
                count_to =0
                count_interjection =0
                                
                text = word_tokenize(row)
                #print (nltk.pos_tag(text))
                pos=nltk.pos_tag(text)                
                pos_noun = ['NN','NNS','NNP','NNPS']
                pos_pronoun = ['PRP','PRP$','WP','Wp$']
                pos_verb = ['VB','VBD','VBG','VBN','VBP','VBZ']
                pos_adjective = ['JJ','JJR','JJS']
                pos_adverb = ['RB','RBR','RBS','WRB']
                pos_determiner = ['DT','PDT','WDT']
                pos_conjuction = ['CC','IN']
                pos_cardinal = ['CD']
                POS_existential = ['EX']
                POS_foreign = ['FW']
                POS_listitem = ['LS']
                POS_modal = ['MD']
                POS_possessive = ['POS']
                POS_particle = ['RP']
                POS_symbol = ['SYM']
                POS_to = ['TO']
                POS_interjection = ['UH']
                
                for word,tag in pos:
                        if tag in pos_noun:
                                count_noun+=1
                        elif tag in pos_pronoun:
                                count_pronoun+=1
                        elif tag in pos_verb:
                                count_verb+=1
                        elif tag in pos_adjective:
                                count_adjective+=1
                        elif tag in pos_adverb:
                                count_adverb+=1
                        elif tag in pos_determiner:
                                count_determiner+=1
                        elif tag in pos_conjuction:
                                count_conjuction+=1
                        elif tag in pos_cardinal:
                                count_cardinal+=1
                        elif tag in POS_existential:
                                count_existential+=1
                        elif tag in POS_foreign:
                                count_foreign+=1
                        elif tag in POS_listitem:
                                count_listitem+=1
                        elif tag in POS_modal:
                                count_modal+=1
                        elif tag in POS_possessive:
                                count_possessive+=1                        
                        elif tag in POS_particle:
                                count_particle+=1
                        elif tag in POS_symbol:
                                count_symbol+=1
                        elif tag in POS_to:
                                count_to+=1
                        elif tag in POS_interjection:
                                count_interjection+=1
                
                Final_Vector[k].append(count_noun)
                Final_Vector[k].append(count_pronoun)
                Final_Vector[k].append(count_verb)
                Final_Vector[k].append(count_adjective)                
                Final_Vector[k].append(count_adverb)
                Final_Vector[k].append(count_determiner)
                Final_Vector[k].append(count_conjuction)
                Final_Vector[k].append(count_cardinal)
                Final_Vector[k].append(count_existential)
                Final_Vector[k].append(count_foreign)
                Final_Vector[k].append(count_listitem)
                Final_Vector[k].append(count_modal)
                Final_Vector[k].append(count_possessive)
                Final_Vector[k].append(count_particle)
                Final_Vector[k].append(count_symbol)
                Final_Vector[k].append(count_to)
                Final_Vector[k].append(count_interjection)
        except:
                print("Exception occur in POS_Stylistics function") 

#This function is used for readabilty score of a text.        
def Readabilty_Score_Features(row,k):  
    try:
        Final_Vector[k].append(textstat.flesch_reading_ease(row))
        Final_Vector[k].append(textstat.gunning_fog(row))
        Final_Vector[k].append(textstat.automated_readability_index(row))
        Final_Vector[k].append(textstat.coleman_liau_index(row))
        Final_Vector[k].append(textstat.syllable_count(row))
        
    except:
        print('Exception occurs in readabilty_score_features function')

#This function is used to calcualte the didderent structural features, such as word count, log word count, punctuation count,
#digits count, and number of captial letters in a text.

def Structural_Features(row,k):
    try:
        cnt_exclm=0
        cnt_dots=0
        cnt_ques_marks=0
        cnt_upper=0
        numbers = sum(c.isdigit() for c in row)
        #Final_Vector[k].append(numbers)
        cap_letters = sum(c.isupper() for c in row)
        #Final_Vector[k].append(cap_letters)
        
        pre_text = re.sub(r'[0-9]+', '', row) # Remove digits
        pre_text= pre_text.replace("!", "") # Remove exclamation marks
        pre_text= pre_text.replace(".", "") # Remove dots
        pre_text= pre_text.replace("?", "") # Remove question marks        
        token = word_tokenize(pre_text) 
        
        #Final_Vector[k].append(len(token))        
        Final_Vector[k].append(math.log(len(token),5))
        print (row)
        for i in range (0, len (row)):
            if row[i] in ('!'):
                cnt_exclm = cnt_exclm + 1                            
            elif row[i] in ('.'):
                cnt_dots = cnt_dots + 1
            elif row[i] in ('?'):
                cnt_ques_marks = cnt_ques_marks + 1                

        print (cnt_exclm,cnt_dots,cnt_ques_marks)      
        Final_Vector[k].append(cnt_exclm)
        Final_Vector[k].append(cnt_dots)
        Final_Vector[k].append(cnt_ques_marks)
    except:
        print("Exception occur in Structural_Features function")

#This function is used for Psycholinguistic_LIWC of a text.

linguistic=['funct','pronoun','ppron','i','we','you','shehe','they','ipron','article','verb',\
            'auxverb','past','present','future','adverb','preps','conj','negate','quant','number','swear']
psychological=['social','family','friend','humans','affect','posemo','negemo','anx','anger','sad','cogmech','insight',\
               'cause','discrep','tentat','certain','inhib','incl','excl','percept','see','hear','feel','bio','body','health',\
               'sexual','ingest','relativ','motion','space','time']
personal = ['work','achieve','leisure','home','money','relig','death']
spoken  = ['assent','nonflu','filler']

parse, category_names = liwc.load_token_parser('LIWC2007_English100131.dic')

def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)

def Psycholinguistic_LIWC(row,k):
    try:        
        print (row)
        cnt_ling=0
        cnt_psyc=0
        cnt_pers=0
        cnt_spk=0
        
        row_tokens = tokenize(row)
        row_counts = Counter(category for token in row_tokens for category in parse(token))
        print(dict(row_counts))
        
        for l in linguistic:
            if l in dict(row_counts):
                ling=dict(row_counts)[l]
                cnt_ling=cnt_ling+ling
        #print(cnt_ling)                
        Final_Vector[k].append(cnt_ling)
        
        for p in psychological:
            if p in dict(row_counts):
                psyc=dict(row_counts)[p]
                cnt_psyc=cnt_psyc+psyc
        #print (cnt_psyc)
        Final_Vector[k].append(cnt_psyc)
        
        for per in personal:
            if per in dict(row_counts):
                pers=dict(row_counts)[per]
                cnt_pers=cnt_pers+pers
        #print (cnt_pers)
        Final_Vector[k].append(cnt_pers)

        for s in spoken:
            if s in dict(row_counts):
                spk=dict(row_counts)[s]
                cnt_spk=cnt_spk+spk
        #print (cnt_spk)
        Final_Vector[k].append(cnt_spk)
    except:
        print("Exception occur in Psycholinguistic_LIWC function")

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
#with open(r'..\Final_Datasets\Founta_18\Founta_Multi_Preprocess.csv', 'r') as my_file:
#with open(r'..\Final_Datasets\Davidson_17\Davidson_Multi_Preprocess.csv', 'r') as my_file:
    
    f = csv.reader(my_file, quoting=csv.QUOTE_ALL)
    reader = next(f) 
    Tag_list = list(f)
    print (Tag_list)
for sublst in Tag_list:
	Final_Vector.append([])        
k=0
for sublst in Tag_list:
  print (sublst)
  label = sublst[1]
  temp=' '.join(map(str, sublst))
  #print (temp)
  Polarity_Score(temp,k)
  Subjectivity_Score(temp,k)
  Positive_Words_Count(temp,k)
  Negative_Words_Count(temp,k)
  Noun_count(temp,k)
  Verb_count(temp,k)
  Adverb_count(temp,k)
  Adjective_count(temp,k)
  Affective_Valence_Score(temp,k)
  Affective_Arousal_Score(temp,k)
  Affective_Dominance_Score(temp,k)
  #POS_Stylistics(temp,k)
  Readabilty_Score_Features(temp,k)
  #Structural_Features(temp,k)
  #Psycholinguistic_LIWC(temp,k)
  Final_Vector[k].append(label)
  print (k)
  k=k+1
print (Final_Vector)

#with open(r'FV_Results\Davidson_Balanced.csv', 'w', newline='') as f:
#with open(r'FV_Results\Founta_Balanced.csv', 'w', newline='') as f:
#with open(r'FV_Results\Gao_Balanced.csv', 'w', newline='') as f:
#with open(r'FV_Results\Roy_Balanced.csv', 'w', newline='') as f:
#with open(r'FV_Results\HateXplain_Balanced.csv', 'w', newline='') as f:
#with open(r'FV_Results\Davidson_Unbalanced.csv', 'w', newline='') as f:
#with open(r'FV_Results\Founta_Unbalanced.csv', 'w', newline='') as f:
#with open(r'FV_Results\Gao_Unbalanced.csv', 'w', newline='') as f:
with open(r'FV_Results\Roy_Unbalanced.csv', 'w', newline='') as f:
#with open(r'FV_Results\HateXplain_Unbalanced.csv', 'w', newline='') as f:
#with open(r'FV_Results\Founta_Multi_Preprocess.csv', 'w', newline='') as f:
#with open(r'FV_Results\Davidson_Multi_Preprocess.csv', 'w', newline='') as f:    
    
    writer = csv.writer(f)
    writer.writerows(Final_Vector)

print (Final_Vector)
print ('Success')


