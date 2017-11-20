#-*- coding: utf-8 -*
import os
import random as rd
import numpy as np
import codecs
import re
import sys
#
def read_tsv(file_name):
    id = []
    context = []
    score = []
    fp = codecs.open(file_name, encoding="UTF8")
    line = fp.readline()

    while True:
        line = fp.readline()
        if not line:
            break
        group = line.strip().split('\t')
        id.append(group[1])
        temp = []
        words = group[2].split(" ")
        for word in words:
            word = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+¡ª¡ª£¡£¬¡££¿?¡¢~@#£¤%ÿÿ&*£¨£©]+", "", word)
            temp.append(word)
        context.append(' '.join(temp))
        score.append(group[-1])

    return id, context, score

#return dic: id, frequent
def word_dict(context):
    dic = {}
    for item in context:
        words = item.split(' ')
        for word in words:
            if not dic.get(word):
                dic[word] = 1
            else:
                dic[word] += 1
    fp = codecs.open("word.txt", "w", encoding='utf-8')
    for key in dic.keys():
        fp.write(str(key) + "\t" + str(dic[key]) + "\n")
    return dic

def read_dic(filename = "word.txt"):
    fp = codecs.open("word.txt", "r", encoding='utf-8')
    id = 0
    dic = {}
    while True:
        line = fp.readline()
        if not line:
            break
        group = line.split('\t')
        dic[group[0]] = id
        id += 1
    return dic




def Batch_Gen(dic, context, score, batch_size, id_begin, id_end, context_size):
    data = np.zeros((batch_size, 2*context_size), dtype = np.int64)
    target = np.zeros((batch_size), dtype = np.int64)
    score_t = np.zeros((batch_size))
    for i in range(batch_size):
        begin = rd.randint(id_begin, id_end-1)
        paragraph = context[begin]
        words = paragraph.split(' ')
        if(len(words) < 2 * context_size + 1):
            i -= 1
            continue
        word_center = rd.randint(context_size, len(words) - context_size-1)
        if dic.get(words[word_center]) == None:
            target[i] = len(dic)
        else:
            target[i] = dic[words[word_center]]
        flag = 0
        for j in range(2 * context_size):
            if j > context_size - 1:
                flag = 1
            if dic.get(words[word_center - context_size + j + flag]) == None:
                data[i, j] = len(dic)
            else:
                data[i,j] = dic[words[word_center - context_size + j + flag]]
        score_t[i] = score[begin]
    return data, target, score_t

def clean_data(input_name, output_fname):
    fp = open(input_name)
    line = fp.readline()
    f_write = open(output_fname, 'w')
    while True:
        line = fp.readline()
        if not line:
            break
        f_write.write(line.decode('cp1252', 'replace').encode('utf-8'))

def Proprecess():
    id, context, score = read_tsv("clean.tsv")
    fre_dic = word_dict(context)

def Test_Batch():
    id, context, score = read_tsv("clean.tsv")
    dic = read_dic()
    data, target, score_t = Batch_Gen(dic, context, score, 32, 0, len(context), 2)



if __name__ == '__main__':
    while True:
        Test_Batch()
















