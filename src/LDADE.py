from __future__ import print_function, division

__author__ = 'amrit'

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
import time
import multiprocessing


learners=[main]
learners_para_dic=[OrderedDict([("n_components",10),("doc_topic_prior",0.1), ("topic_word_prior",0.01)])]
learners_para_bounds=[[(10,100), (0.1,1), (0.01,1)]]
learners_para_categories=[[ "integer", "continuous", "continuous"]]
ROOT=os.getcwd()
files=["pitsA", "pitsB", "pitsC", "pitsD", "pitsE", "pitsF"]
MLS=[DT,RF, SVM,  FFT1]
MLS_para_dic=[OrderedDict([("min_samples_split",2),("min_impurity_decrease",0.0), ("max_depth",None),
                               ("min_samples_leaf", 1)]), OrderedDict([("min_samples_split",2),
                                ("max_leaf_nodes",None), ("min_samples_leaf",1), ("min_impurity_decrease",0.0),("n_estimators",10)]),
                   OrderedDict([("C", 1.0), ("kernel", 'linear'),
                                ("degree", 3)]), OrderedDict()]

metrics = ['recall', 'precision', 'f1']


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

def _test(res=''):
    seed(1)
    np.random.seed(1)
    path=ROOT+"/../data/preprocessed/"+res+".txt"
    raw_data, labels=readfile1(path)
    temp={}

    for m in metrics:
        for i in range(5):
            ranges=list(range(labels.shape[0]))
            shuffle(ranges)
            raw_data=raw_data[ranges]
            labels=labels[ranges]
            start_time=time.time()
            de = DE(Goal="Max", GEN=5, NP=10,termination="Early")
            v, _ = de.solve(learners[0], OrderedDict(learners_para_dic[0]),
                            learners_para_bounds[0], learners_para_categories[0],
                            file=res, term=7, data_samples=raw_data)
            corpus,_=LDA_(raw_data,**v.ind)
            end_time = time.time()-start_time

            skf = StratifiedKFold(n_splits=5)
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
                        temp[le.__name__]['times']=[]
                    else:
                        temp[le.__name__]['times'].append(end_time1+end_time)
                    if 'features' not in temp[le.__name__]:
                        temp[le.__name__]['features'] = []
                    else:
                        temp[le.__name__]['features'].append(val[1])
                    print(temp)

    with open('../dump/LDADE' +res+ '_1.pickle', 'wb') as handle:
        pickle.dump(temp, handle)


def run(res=''):
    if res == "pits":
        res = ['pitsA', 'pitsB', 'pitsC', 'pitsD', 'pitsE', 'pitsF']
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size,
                                maxtasksperchild=4,)
    for r in res:
        pool.apply_async(_test, args=(r, ))

    '''
    procs = []
    proc = Process(target=_test, args=(5,))
    if res == "pits":
        res = ['pitsA', 'pitsB', 'pitsC', 'pitsD', 'pitsE', 'pitsF']
    for r in res:
        proc = Process(target=_test, args=(r,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
    '''


if __name__ == '__main__':
    eval(cmd ())
