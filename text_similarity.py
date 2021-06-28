# -*- coding:utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text1= "text1.txt" # file path
text2= "text2.txt"
Document1 = ''
Document2 = ''

f1 = open(text1,"r")
lines = f1.readlines()
for line1 in lines:
    Document1 = Document1 + line1

f2 = open(text2,"r")
lines = f2.readlines()
for line2 in lines:
    Document2 = Document2 + line2

print('Document 1:',Document1)
print('Document 2:',Document2)


corpus = [Document1,Document2]

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
            'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who',
            'whom', 'this', 'that', 'these', 'those', 'am',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'of', 'at',
            'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each',
            'few', 'more', 'most', 'other', 'some', 'such',
            'nor', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don',
            'should', 'now', ',','.','sir','much','ms','?','!','n','and']

cvec = CountVectorizer(min_df=0,stop_words=None,token_pattern = r'[^\s]+') # stop_words = None: not use stop words
x_train_c = cvec.fit_transform(corpus)
features = cvec.get_feature_names()
print('c x_train',x_train_c.toarray())
print(x_train_c.shape)
print('features:',features)

s = cosine_similarity(x_train_c[0], x_train_c[1])
print('similarity:',s)