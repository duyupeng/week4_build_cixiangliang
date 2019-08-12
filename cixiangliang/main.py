#!/usr/bin/env python
#-*- coding: utf-8 -*-
import re
import jieba
import os
import nltk
from gensim.models import word2vec
from sklearn.manifold import TSNE

import matplotlib; matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from gensim.models.word2vec import LineSentence

from gensim.models import Word2Vec

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题



out_path=os.path.join(os.getcwd(),"output")
# in_path="C:\\Users\\zuoti\\PycharmProject\\cixiangliang\\AA\\result_00.txt"

# in_path=os.path.join(os.getcwd(),"AA\\result_00")
in_path=os.path.join(os.getcwd())
print('1',in_path)
space=re.compile('\s')

def parse_zhwiki(read__path,save_path):
    # output_path = out_path

    path=["\AA\\result_00","\AA\\result_01","\AA\\result_00"]
    for path_ in path:
        read_file_path=read__path+'{}'.format(path_)
        file=open(read_file_path,'r',encoding='utf-8')
        content_line=file.readline()
    # print(content_line)

        while content_line:
            regex_str ="[^<doc.*>$]|[^</doc>$]"
            # content_line='哲学（）是研究普遍的、基础问题的学科，包括存在、知识、价值、理智、心灵、语言等领域。哲学与其他学科不同之处在于哲学有独特之思考方式，例如批判的方式'
            match_obj = re.match(regex_str,content_line)
            if len(content_line)>0:
                if match_obj:
                    content_line=space.sub('',content_line)
                    words = jieba.cut(content_line)
                    line=' '.join(list(words))
                    if not out_path:
                        os.makedirs(out_path)
                    # save_path=save_path+'\cut_out'
                    with open(save_path+'\cut_out','a',encoding="utf-8") as f:
                        f.write(line+'\n')
            content_line=file.readline()
            # print('22',content_line)


# parse_zhwiki(in_path,out_path)


def build_corpus(cut_out_file_path):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    with open(cut_out_file_path,'r',encoding="utf-8") as f:
        data=f.readlines()
    corpus = []
    stops=re.compile('\n|、|。| ，|《|》|「|」|\[|\]|（|）|“|”|·|')
    # print(data[0:100])
    # data=data[0:10000]

    for sentence in data:
        data_new=stops.sub('',sentence)
        # print(data_new)
        if data_new !='':
            word_list = data_new.strip().split(" ")
            word_list_=[i for i in word_list if i!='']
        corpus.append(word_list_)

    return corpus


# cut_out_file_path=os.path.join(os.getcwd(),'output/cut_out')
# # build_corpus(cut_out_file_path)
# corpus=build_corpus(cut_out_file_path)
# model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)
# # print(model.wv['数学'])


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()



cut_out_file_path=os.path.join(os.getcwd(),'output/cut_out')
# build_corpus(cut_out_file_path)
corpus=build_corpus(cut_out_file_path)
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)
# print(model.wv['数学'])
print('model训练完成')
tsne_plot(model)


# cut_out_file_path=os.path.join(os.getcwd(),'output/cut_out')
# wiki_news = open(cut_out_file_path, 'r',encoding='utf-8')
# model = Word2Vec(LineSentence(wiki_news), sg=0,size=200, window=10, min_count=500, workers=6)
# print('model训练完成')
# tsne_plot(model)