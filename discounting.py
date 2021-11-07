#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 10:28:06 2021

@author: zzh
"""
import nltk
import pickle
from nltk import FreqDist
from nltk.util import ngrams
import numpy as np
from nltk.tokenize import (TreebankWordTokenizer,
                           word_tokenize,
                           wordpunct_tokenize,
                           TweetTokenizer,
                           MWETokenizer)
sentence = open('train_set.txt').read()
#分词
tokenizer = TweetTokenizer()
tokenizer = TreebankWordTokenizer()
data = tokenizer.tokenize(sentence)
fdist_2 = FreqDist()
n=100
all_counts = dict()
Prob_1={}
Prob_2={}
Prob_3={}
Prob_4={}
Prob_5={}
#统计词频与概率计算
for key in all_counts[1].keys():
    Prob_1[key]=all_counts[1].freq(key)
min_value=min(Prob_1.values())
#将低频词转化为UNK
Prob_1['UNK']=0
for k in list(Prob_1.keys()):
    if Prob_1[k]==min_value:
        Prob_1.pop(k)
        Prob_1['UNK']+=min_value   
        
for key in all_counts[2].keys():
    Prob_2[key]=all_counts[2].freq(key)
min_value=min(Prob_2.values())
Prob_2['UNK']=0
for k in list(Prob_2.keys()):
    if Prob_2[k]==min_value:
        Prob_2.pop(k)
        Prob_2['UNK']+=min_value
        
for key in all_counts[3].keys():
    Prob_3[key]=all_counts[3].freq(key)
min_value=min(Prob_3.values())
Prob_3['UNK']=0
for k in list(Prob_3.keys()):
    if Prob_3[k]==min_value:
        Prob_3.pop(k)
        Prob_3['UNK']+=min_value
        
for key in all_counts[4].keys():
    Prob_4[key]=all_counts[4].freq(key)
min_value=min(Prob_4.values())
Prob_4['UNK']=0
for k in list(Prob_4.keys()):
    if Prob_4[k]==min_value:
        Prob_4.pop(k)
        Prob_4['UNK']+=min_value
       
for key in all_counts[5].keys():
    Prob_5[key]=all_counts[5].freq(key)
min_value=min(Prob_5.values())
Prob_5['UNK']=0
for k in list(Prob_5.keys()):
    if Prob_5[k]==min_value:
        Prob_5.pop(k)
        Prob_5['UNK']+=min_value
Prob=[Prob_1,Prob_2,Prob_3,Prob_4,Prob_5]
#求出最低的几个频率并统计数量
min_1_list=[]
min_2_list=[]
min_3_list=[]
min_4_list=[]
min_5_list=[]
for i in range(5):
    Probability = Prob[i]
    min_1=1
    min_2=1
    min_3=1
    min_4=1
    for k,v in Probability.items():
        if v<=min_1:
            min_1=v
        else:
            if v<=min_2:
                min_2=v
            else:
                if v<=min_3:
                    min_3=v
                else:
                    if v<=min_4:
                        min_4=v
    if i == 0:
        min_1_list.append(min_1)
        min_1_list.append(min_2)
        min_1_list.append(min_3)
        min_1_list.append(min_4)
    if i == 1:
        min_2_list.append(min_1)
        min_2_list.append(min_2)
        min_2_list.append(min_3)
        min_2_list.append(min_4)
    if i == 2:
        min_3_list.append(min_1)
        min_3_list.append(min_2)
        min_3_list.append(min_3)
        min_3_list.append(min_4)
    if i == 3:
        min_4_list.append(min_1)
        min_4_list.append(min_2)
        min_4_list.append(min_3)
        min_4_list.append(min_4)
    if i == 4:
        min_5_list.append(min_1)
        min_5_list.append(min_2)
        min_5_list.append(min_3)
        min_5_list.append(min_4)
min_list=[min_1_list,min_2_list,min_3_list,min_4_list,min_5_list]   
alpha=0.5
#dicounting算法的实现
for i in range(5):
    freq_1=0
    freq_2=0
    freq_3=0
    freq_4=0
    counter_1=0
    counter_2=0
    counter_3=0
    counter_4=0
    for k, v in Prob[i].items():
        if v == min_list[i][0]:
            freq_1+=min_1_list[0]
            counter_1+=1
        if v == min_list[i][1]:
            freq_2+=min_1_list[1]
            counter_2+=1
        if v == min_list[i][2]:
            freq_3+=min_1_list[2]
            counter_3+=1
        if v == min_list[i][3]:
            freq_4+=min_1_list[3]
            counter_4+=1
    temp=alpha*freq_4
    freq_4=(1-alpha)*freq_4
    freq_3+=temp
    temp=alpha*freq_3
    freq_3=(1-alpha)*freq_3
    freq_2+=temp
    temp=alpha*freq_2
    freq_2=(1-alpha)*freq_2
    freq_1+=temp
    freq_each_1=freq_1/counter_1
    freq_each_2=freq_2/counter_2
    freq_each_3=freq_3/counter_3
    freq_each_4=freq_4/counter_4
    for k, v in Prob[i].items():
     if v == min_list[i][0]:
         Prob[i][k]=freq_each_1
     if v == min_list[i][1]:
         Prob[i][k]=freq_each_2
     if v == min_list[i][2]:
         Prob[i][k]=freq_each_3
     if v == min_list[i][3]:
         Prob[i][k]=freq_each_4     
#测试集的加载与分词
sentence = open('test_set.txt').read()  
tokenizer = TreebankWordTokenizer()
data = tokenizer.tokenize(sentence)
gram_1= ngrams(data, 1)
gram_2= ngrams(data, 2)
gram_3= ngrams(data, 3)
gram_4= ngrams(data, 4)
gram_5= ngrams(data, 5)
K=1700687
Prob_list=[]
#PPL的计算，可以将gram_3变量替换成其他gram计算（同时要改变Prob对应的编号）
for ngram in gram_3:
    if ngram in Prob[2].keys():
        Prob_list.append(pow(1/(Prob[2][ngram]),1/K))
    else:
        Prob_list.append(pow(1/(Prob[2]['UNK']),1/K))

product=1
for i in Prob_list:
    product = product*i

print(product)




