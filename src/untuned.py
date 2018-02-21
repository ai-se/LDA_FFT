from __future__ import print_function, division

__author__ = 'amrit'

import sys
from demo import cmd
sys.dont_write_bytecode = True
from ldagibbs import *
from collections import OrderedDict
import os
from collections import Counter
from featurization import *
from ML import DT, SVM, RF, FFT1
from sklearn.model_selection import StratifiedKFold
import pickle

learners=[main]
ROOT=os.getcwd()
files=["pitsA", "pitsB", "pitsC", "pitsD", "pitsE", "pitsF"]
MLS=[DT,RF, SVM,  FFT1]
MLS_para_dic=[OrderedDict([("min_samples_split",2),("min_impurity_decrease",0.0), ("max_depth",None),
                               ("min_samples_leaf", 1)]), OrderedDict([("min_samples_split",2),
                                ("max_leaf_nodes",None), ("min_samples_leaf",1), ("min_impurity_decrease",0.0),("n_estimators",10)]),
                   OrderedDict([("C", 1.0), ("kernel", 'linear'),
                                ("degree", 3)]), OrderedDict()]

metrics=['accuracy','recall','precision','false_alarm']
features=[TF, TFIDF, HASHING]

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
    key=max(count.iteritems(), key=operator.itemgetter(1))[0]

    labels=map(lambda x: 1 if x == key else 0, labels)
    return np.array(dict), np.array(labels)

def _test(res=''):
    seed(1)
    np.random.seed(1)
    path=ROOT+"/../data/preprocessed/"+res+".txt"
    raw_data,labels=readfile1(path)
    temp={}

    for i in range(5):
        ranges=range(len(labels))
        shuffle(ranges)
        raw_data=raw_data[ranges]
        labels=labels[ranges]

        for fea in features:
            if fea.__name__ not in temp:
                temp[fea.__name__] = {}
            corpus,_=fea(raw_data)

            skf = StratifiedKFold(n_splits=5)
            for train_index, test_index in skf.split(corpus, labels):
                train_data, train_labels = corpus[train_index], labels[train_index]
                test_data, test_labels = corpus[test_index], labels[test_index]

                for j, le in enumerate(MLS):
                    if le.__name__ not in temp[fea.__name__]:
                        temp[fea.__name__][le.__name__]={}

                    _,val=MLS[j](MLS_para_dic[j], train_data, train_labels, test_data, test_labels, 'recall')
                    for m in metrics:
                        if m not in temp[fea.__name__][le.__name__]:
                            temp[fea.__name__][le.__name__][m]=[]
                        temp[fea.__name__][le.__name__][m].append(val[0][m])
    print(temp)
    with open('../dump/untuned' +res+ '.pickle', 'wb') as handle:
        pickle.dump(temp, handle)

if __name__ == '__main__':
    eval(cmd ())