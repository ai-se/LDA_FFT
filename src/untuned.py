from __future__ import print_function, division

__author__ = 'amrit and huy'

import sys
from demo import cmd
#sys.dont_write_bytecode = True
from collections import OrderedDict
import os
from collections import Counter
from featurization import *
from ML import DT, SVM, RF, FFT1
from sklearn.model_selection import StratifiedKFold
import pickle
import numpy as np
from random import seed, shuffle
import time
import multiprocessing as mp


ROOT=os.getcwd()
files=["pitsA", "pitsB", "pitsC", "pitsD", "pitsE", "pitsF"]
MLS=[SVM]#,  FFT1]
#MLS_para_dic=[OrderedDict([("min_samples_split",2),("min_impurity_decrease",0.0), ("max_depth",None),
#                               ("min_samples_leaf", 1)]), OrderedDict([("min_samples_split",2),
#                                ("max_leaf_nodes",None), ("min_samples_leaf",1), ("min_impurity_decrease",0.0),("n_estimators",10)]),
MLS_para_dic = [OrderedDict([("C", 1.0), ("kernel", 'linear'), ("degree", 3)]),
                OrderedDict()]

metrics=['recall','precision']
features=[TFIDF, HASHING]


def readfile1(filename=''):
    dict = []
    labels=[]
    with open(filename, 'r') as f:
        for doc in f.readlines():
            try:
                row = doc.lower().split(">>>")
                dict.append(row[0].strip())
                labels.append(row[1].strip())
            except:
                pass
    count=Counter(labels)
    import operator
    key=max(count.items(), key=operator.itemgetter(1))[0]

    labels=list(map(lambda x: 1 if x == key else 0, labels))
    return np.array(dict), np.array(labels)


def mining_parallel(MLS, corpus, labels, train_index, test_index, end_time, m):
    train_data, train_labels = corpus[train_index], labels[train_index]
    test_data, test_labels = corpus[test_index], labels[test_index]
    result = {}
    for j, le in enumerate(MLS):
        result[le.__name__] = {}
        start_time1 = time.time()
        _, val = MLS[j](MLS_para_dic[j], train_data, train_labels, test_data, test_labels, m)
        end_time1 = time.time() - start_time1
        result[le.__name__]['perf'] = val[0][m]
        result[le.__name__]['times'] = end_time1 + end_time
        result[le.__name__]['features'] = val[1]
    return result


def parallel_test(res=''):
    seed(1)
    np.random.seed(1)
    path=ROOT+"/../data/pits_preprocessed/"+res+".txt"
    raw_data,labels=readfile1(path)
    temp={}

    for m in metrics:
        temp[m] = {}
        for i in range(5):
            ranges=list(range(len(labels)))
            shuffle(ranges)
            raw_data=raw_data[ranges]
            labels=labels[ranges]
            for fea in features:
                temp[m][fea.__name__] = {}
                start_time = time.time()
                corpus,_=fea(raw_data)
                end_time = time.time() - start_time
                skf = StratifiedKFold(n_splits=5)
                for le in MLS:
                    temp[m][fea.__name__][le.__name__] = {}
                results = []
                pool = mp.Pool()
                for train_index, test_index in skf.split(corpus, labels):
                    #results.append(mining_parallel(MLS, corpus, labels, train_index, test_index, end_time, m))
                    pool.apply_async(mining_parallel, args=(MLS, corpus, labels, train_index, test_index, end_time, m,), callback=results.append)
                pool.close()
                pool.join()
                for res in results:
                    for k in res.keys():
                        for sub_k in res[k].keys():
                            if sub_k not in temp[m][fea.__name__][k].keys():
                                temp[m][fea.__name__][k][sub_k] = [res[k][sub_k]]
                            else:
                                temp[m][fea.__name__][k][sub_k].append(res[k][sub_k])

    #import pdb
    #pdb.set_trace()
    with open('../dump/untuned' +res+ '_1.pickle', 'wb') as handle:
        pickle.dump(temp, handle)


def run(res=''):
    if res == "pits":
        res = ['pitsA', 'pitsB', 'pitsC', 'pitsD', 'pitsE', 'pitsF']
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size,
                                maxtasksperchild=2,)
    for r in res:
        pool.apply_async(_test, args=(r, ))


if __name__ == '__main__':
    eval(cmd())
