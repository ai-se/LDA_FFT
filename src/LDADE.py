from __future__ import print_function, division

__author__ = 'amrit and huy'

import sys
from demo import cmd
from DE import DE
from ldavem import *
from collections import OrderedDict
import os
from collections import Counter
from featurization import *
from ML import DT, SVM, RF, FFT1
from sklearn.model_selection import StratifiedKFold
import pickle
import operator
import time
import pandas as pd
import multiprocessing as mp


learners=[main]
learners_para_dic=[OrderedDict([("n_components",10),("doc_topic_prior",0.1), ("topic_word_prior",0.01)])]
learners_para_bounds=[[(10,100), (0.1,1), (0.01,1)]]
learners_para_categories=[[ "integer", "continuous", "continuous"]]
ROOT=os.getcwd()
files=["pitsA", "pitsB", "pitsC", "pitsD", "pitsE", "pitsF"]
MLS=[SVM]
#MLS_para_dic=[OrderedDict([("min_samples_split",2),("min_impurity_decrease",0.0), ("max_depth",None),
#                               ("min_samples_leaf", 1)]), OrderedDict([("min_samples_split",2),
#                                ("max_leaf_nodes",None), ("min_samples_leaf",1), #("min_impurity_decrease",0.0),("n_estimators",10)]),

MLS_para_dic=[OrderedDict([("C", 1.0), ("kernel", 'linear'), ("degree", 3)]), OrderedDict()]
metrics = ['precision', 'recall', 'f1']


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
    key = max(count.items(), key=operator.itemgetter(1))[0]
    labels = list(map(lambda x: 1 if x == key else 0, labels))
    return np.array(dict), np.array(labels)


def readfile(filename=''):
    df = pd.read_csv(filename, delimiter=";", index_col=False)

    #print(df)
    text_dict_raw = df['texts'].values.tolist()
    labels_raw = df['labels'].values.tolist()
    labels = np.array(labels_raw)
    text_dict = np.array([x.strip() for x in text_dict_raw])
    return text_dict, labels



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


def parallel_test(fname=''):
    print(fname, "LDADE")
    seed(1)
    np.random.seed(1)
    raw_p = fname.split("_")
    folder = ROOT + "/../data/" + ("%s_preprocessed" % raw_p[0])
    path = folder + "/" + fname + ".csv"
    raw_data,labels=readfile(path)

    temp={}

    for m in metrics:
        temp[m] = {}
        for le in MLS:
            temp[m][le.__name__] = {}
        for i in range(5):
            print(m, i)
            ranges=list(range(labels.shape[0]))
            shuffle(ranges)
            raw_data=raw_data[ranges]
            labels=labels[ranges]
            start_time=time.time()
            de = DE(Goal="Max", GEN=1, NP=10,termination="Early")
            v, _ = de.solve(learners[0], OrderedDict(learners_para_dic[0]),
                            learners_para_bounds[0], learners_para_categories[0],
                            file=res, term=7, data_samples=raw_data)
            corpus,_=LDA_(raw_data,**v.ind)
            end_time = time.time()-start_time

            skf = StratifiedKFold(n_splits=5)
            results = []
            pool = mp.Pool()
            start_running = time.time()
            for train_index, test_index in skf.split(corpus, labels):
                pool.apply_async(mining_parallel,
                                 args=(MLS, corpus, labels, train_index, test_index, end_time, m,),
                                 callback=results.append)
            pool.close()
            pool.join()
            print("time to run this", time.time() - start_running)
            for res in results:
                for k in res.keys():
                    for sub_k in res[k].keys():
                        if sub_k not in temp[m][k].keys():
                            temp[m][k][sub_k] = [res[k][sub_k]]
                        else:
                            temp[m][k][sub_k].append(res[k][sub_k])
        # import pdb
        # pdb.set_trace()
        with open('../dump/LDADE_' + fname + '_1.pickle', 'wb') as handle:
            pickle.dump(temp, handle)



def _test(res=''):
    seed(1)
    np.random.seed(1)
    path=ROOT+"/../data/pits_preprocessed/"+res+".txt"
    raw_data, labels=readfile1(path)
    temp={}

    for m in metrics:
        for i in range(2):
            print(m, i)
            ranges=list(range(labels.shape[0]))
            shuffle(ranges)
            raw_data=raw_data[ranges]
            labels=labels[ranges]
            start_time=time.time()
            de = DE(Goal="Max", GEN=1, NP=10,termination="Early")
            v, _ = de.solve(learners[0], OrderedDict(learners_para_dic[0]),
                            learners_para_bounds[0], learners_para_categories[0],
                            file=res, term=7, data_samples=raw_data)
            corpus,_=LDA_(raw_data,**v.ind)
            end_time = time.time()-start_time

            skf = StratifiedKFold(n_splits=5)
            start_running = time.time()
            for train_index, test_index in skf.split(corpus, labels):
                train_data, train_labels = corpus[train_index], labels[train_index]
                test_data, test_labels = corpus[test_index], labels[test_index]

                for j, le in enumerate(MLS):
                    if le.__name__ not in temp:
                        temp[le.__name__]={}
                    start_time1 = time.time()
                    _, val = MLS[j](MLS_para_dic[j], train_data, train_labels, test_data, test_labels, m)
                    end_time1=time.time()-start_time1
                    if m not in temp[le.__name__]:
                        temp[le.__name__][m]=[]
                    temp[le.__name__][m].append(val[0][m])
                    if 'times' not in temp[le.__name__]:
                        temp[le.__name__]['times']=[end_time1+end_time]
                    else:
                        temp[le.__name__]['times'].append(end_time1+end_time)
                    if 'features' not in temp[le.__name__]:
                        temp[le.__name__]['features'] = [val[1]]
                    else:
                        temp[le.__name__]['features'].append(val[1])
                    #print(temp)
            print("time to run this", time.time() - start_running)
    import pdb
    pdb.set_trace()
    with open('../dump/LDADE' +res+ '_1.pickle', 'wb') as handle:
        pickle.dump(temp, handle)



if __name__ == '__main__':
    eval(cmd ())
