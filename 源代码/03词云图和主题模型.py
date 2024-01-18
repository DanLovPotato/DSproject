# -*- coding: utf-8 -*-
"""
Created on Fri 

@author: 
"""
##导入库
import pandas as pd
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import numpy as np
from gensim import corpora, models
import gensim

data = pd.read_csv("jobs_with_description.csv")
job = [x for x in data['职位描述'].values]

job_txt=''
for x in job:
    job_txt+=str(x)
    

with open('Stopword.txt',"r",encoding='utf-8') as f:
    stopwords = f.read()
stopwords = stopwords.split('\n')

jieba.load_userdict("userdict.txt")

wordlist = jieba.lcut(job_txt)
wordlist =[x for x in wordlist if x not in stopwords]
wl = " ".join(wordlist)

from collections import Counter 
def get_words(txt): 
     c = Counter() 
     for x in txt: 
         if len(x)>1: 
             c[x] += 1 
     print('常用词频度统计结果') 
     for (k,v) in c.most_common(50): 
         print('%s,%d' % (k, v)) 
get_words(wordlist)
##绘制词云
image = Image.open('1.jpg') 
img = np.array(image) 
produCloud=WordCloud(background_color="white",font_path='C:/windows/Fonts/STZHONGS.TTF', mask=img,max_words=200, min_font_size=20,max_font_size=80,random_state=50)
myword = produCloud.generate(wl)
plt.imshow(myword)
plt.axis("off")
plt.show()
produCloud.to_file('ciyu.jpg') 

##主题模型
data = pd.read_csv("jobs_with_description.csv")
job = [x for x in data.dropna()['职位描述'].values]
jobs = [jieba.lcut(x) for x in job]
jobs = [[x.strip() for x in y if x not in stopwords] for y in jobs]
jobs = [list(filter(None,x)) for x in jobs]

dictionary = corpora.Dictionary(jobs)
corpus = [dictionary.doc2bow(text) for text in jobs]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lda = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=5)
lda.print_topics(num_topics=5, num_words=10)






