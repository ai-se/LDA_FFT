from __future__ import print_function, division

__author__ = 'amrit and huy'

import sys
from demo import cmd

#sys.dont_write_bytecode = True
from collections import OrderedDict
import os
from random import seed, shuffle
import numpy as np
from collections import Counter
from featurization import *
from ML import DT, SVM, RF, FFT1
from sklearn.model_selection import StratifiedKFold
import pickle
import time
import operator
import pandas as pd
import multiprocessing as mp

ROOT = os.getcwd()
domains = ['pits', 'ciscripts']
files = ["pitsA", "pitsB", "pitsC", "pitsD", "pitsE", "pitsF"]
#MLS = [DT, RF, SVM,FFT1]

#MLS_para_dic = [OrderedDict([("min_samples_split", 2), ("min_impurity_decrease", 0.0), ("max_depth", None),
#                             ("min_samples_leaf", 1)]), OrderedDict([("min_samples_split", 2),
#                                                                     ("max_leaf_nodes", None), ("min_samples_leaf", 1),
#                                                                     ("min_impurity_decrease", 0.0),
#                                                                     ("n_estimators", 10)]),
MLS = [FFT1]
MLS_para_dic = [OrderedDict([("C", 1.0), ("kernel", 'linear'),
                             ("degree", 3)]), OrderedDict()]

metrics = ['precision', 'recall', 'f1']
features = ['10']
#metrics = ['recall']
#features = ['10']

def readfile1(filename=''):
    text_dict = []
    labels = []
    print(filename)
    with open(filename, 'r') as f:
        for doc in f.readlines():
            try:
                row = doc.lower().split(">>>")
                text_dict.append(row[0].strip())
                labels.append(row[1].strip())
            except:
                pass
    count = Counter(labels)
    import operator
    #print(count, labels)
    key = max(count.items(), key=operator.itemgetter(1))[0]
    print(max(count.items(), key=operator.itemgetter(1)))
    update_labels = list(map(lambda x: 1 if x == key else 0, labels))
    return np.array(text_dict), np.array(update_labels)


def readfile(filename=''):
    df = pd.read_csv(filename, delimiter=";", index_col=False)

    #print(df)
    text_dict_raw = df['texts'].values.tolist()
    labels_raw = df['labels'].values.tolist()
    labels = np.array(labels_raw)
    text_dict = np.array([x.strip() for x in text_dict_raw])
    return text_dict, labels


def _test(res=''):
    seed(1)
    np.random.seed(1)
    path = ROOT + "/../data/pits_preprocessed/" + res + ".csv"
    raw_data, labels = readfile1(path)
    temp = {}
    for m in metrics:
        #print(m)
        for i in range(2):
            #print("repeat", i, len(features))
            ranges = list(range(labels.shape[0]))
            shuffle(ranges)
            raw_data = raw_data[ranges]
            labels = labels[ranges]
            for fea in features:
                if fea not in temp:
                    temp[fea] = {}
                start_time = time.time()
                corpus, _ = LDA_(raw_data, n_components=int(fea))
                end_time = time.time() - start_time
                skf = StratifiedKFold(n_splits=5)
                start_running = time.time()
                for train_index, test_index in skf.split(corpus, labels):
                    train_data, train_labels = corpus[train_index], labels[train_index]
                    test_data, test_labels = corpus[test_index], labels[test_index]
                    for j, le in enumerate(MLS):
                        if le.__name__ not in temp[fea]:
                            temp[fea][le.__name__] = {}
                        start_time1 = time.time()

                        _,val = MLS[j](MLS_para_dic[j], train_data, train_labels, test_data, test_labels, m)
                        end_time1 = time.time() - start_time1
                        if m not in temp[fea][le.__name__]:
                            temp[fea][le.__name__][m] = []
                        temp[fea][le.__name__][m].append(val[0][m])
                        if 'times' not in temp[fea][le.__name__]:
                            temp[fea][le.__name__]['times']=[end_time1 + end_time]
                        else:
                            temp[fea][le.__name__]['times'].append(end_time1 + end_time)
                        if 'features' not in temp[fea][le.__name__]:
                            temp[fea][le.__name__]['features']=[val[1]]
                        else:
                            temp[fea][le.__name__]['features'].append(val[1])
                print("time to run this", time.time() - start_running)

    #import pdb
    #pdb.set_trace()
    with open('../dump/LDA' + res + '_1.pickle', 'wb') as handle:
        pickle.dump(temp, handle)


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


def getting_files():
    file_names = []
    for _, _, files in os.walk("../data/raw/"):
        for file in files:
            if file.startswith("pits"):
                file_names.append(file)



def parallel_test(fname=''):
    seed(1)
    np.random.seed(1)
    raw_p = fname.split("_")
    folder = ROOT + "/../data/" + ("%s_preprocessed" % raw_p[0])
    path = folder + "/" + fname + ".csv"
    raw_data, labels = readfile(path)
    temp={}

    for m in metrics:
        temp[m] = {}
        for i in range(5):
            print(m, i)
            ranges=list(range(len(labels)))
            shuffle(ranges)
            raw_data=raw_data[ranges]
            labels=labels[ranges]
            for fea in features:
                if fea not in temp[m]:
                    temp[m][fea] = {}
                    for le in MLS:
                        temp[m][fea][le.__name__] = {}
                start_time = time.time()
                corpus, _ = LDA_(raw_data, n_components=int(fea))
                end_time = time.time() - start_time
                skf = StratifiedKFold(n_splits=5)
                results = []
                pool = mp.Pool()
                start_running = time.time()
                for train_index, test_index in skf.split(corpus, labels):
                    #results.append(mining_parallel(MLS, corpus, labels, train_index, test_index, end_time, m))
                    pool.apply_async(mining_parallel,
                                     args=(MLS, corpus, labels, train_index, test_index, end_time, m,),
                                     callback=results.append)
                pool.close()
                pool.join()
                print("time to run this", time.time() - start_running)
                for res in results:
                    for k in res.keys():
                        for sub_k in res[k].keys():
                            if sub_k not in temp[m][fea][k].keys():
                                temp[m][fea][k][sub_k] = [res[k][sub_k]]
                            else:
                                temp[m][fea][k][sub_k].append(res[k][sub_k])
        #import pdb
        #pdb.set_trace()
        with open('../dump/LDA_' + fname + '_1.pickle', 'wb') as handle:
            pickle.dump(temp, handle)


def testing(res=''):
    path = ROOT + "/../data/pits_preprocessed/" + res + ".csv"
    path1 = ROOT + "/../data/pits_preprocessed/" + res + ".txt"
    a1, a2 = readfile(path)
    b1, b2 = readfile1(path1)
    #print(a2, b2)
    if list(a1) == list(b1):
        print('yes')
    if list(a2) == list(b2):
        print("yes")
    else:
        print("no")


def testing_path(res=''):
    raw_p = res.split("_")
    folder = ROOT + "/../data/" + ("%s_preprocessed" % raw_p[0])
    path = folder + "/" + res + ".csv"
    df = pd.read_csv(path, delimiter=";", index_col=False)
    print(df)

if __name__ == '__main__':
    eval(cmd ())


