from __future__ import print_function, division

__author__ = 'amrit'

import sys

sys.dont_write_bytecode = True
from sklearn.feature_extraction.text import HashingVectorizer,CountVectorizer,TfidfVectorizer
import lda

def TF(doc):
    vect= CountVectorizer(max_df=0.95, min_df=2,)
    X= vect.fit_transform(doc)
    return X.A,vect.get_feature_names()

def TFIDF(doc):
    vect= TfidfVectorizer()
    X= vect.fit_transform(doc)
    return X.A, vect.get_feature_names()

def HASHING(doc):
    vect= HashingVectorizer(n_features=1000)
    X= vect.fit_transform(doc)
    return X.A, ''

def LDA_(doc,**k):
    corpus, vocab=TF(doc)
    model=lda.LDA(n_iter=1000,**k)
    model.fit(corpus)
    return model.doc_topic_, vocab