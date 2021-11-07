#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 08:59:24 2021

@author: zzh
"""
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.tokenize import (TreebankWordTokenizer,
                           word_tokenize,
                           wordpunct_tokenize,
                           TweetTokenizer,
                           MWETokenizer)

sentence = open('train_set.txt').read()
tokenizer = TreebankWordTokenizer()
sentence = tokenizer.tokenize(sentence)
paddedLine = [list(pad_both_ends(sentence, n=3))]
train, vocab = padded_everygram_pipeline(3, paddedLine)
from nltk.lm import MLE, KneserNeyInterpolated,Laplace, WittenBellInterpolated
lm = Laplace(3)
lm.fit(train, vocab)
print(lm.vocab)
test= open('train_set.txt').read()
tokenizer = TweetTokenizer()
tokenizer = TreebankWordTokenizer()
test = tokenizer.tokenize(test)
gram_3= ngrams(test,3)
print(lm.perplexity(gram_3))
