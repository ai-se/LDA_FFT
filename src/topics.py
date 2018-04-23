from __future__ import print_function, division

__author__ = 'amrit'

from LDADE import *
files=["pitsA", "pitsB", "pitsC", "pitsD", "pitsE", "pitsF"]
def get_top_words(model, path1, feature_names, n_top_words, i=0, file1=''):
    topics = []
    fo = open(path1+file1+'.txt', 'w+')
    fo.write("Run: " + str(i) + "\n")
    for topic_idx, topic in enumerate(model.components_):
        str1 = ''
        fo.write("Topic " + str(topic_idx) + ": ")
        for j in topic.argsort()[:-n_top_words - 1:-1]:
            str1 += feature_names[j] + " "
            fo.write(feature_names[j] + " ")
        str1=str(str1.encode('ascii', 'ignore'))
        topics.append(str1)
        fo.write("\n")
    fo.close()
    return topics

seed(1)
np.random.seed(1)
path1=ROOT+"/../results/"
for res in files:
    print(res)
    path=ROOT+"/../data/preprocessed/"+res+".txt"
    raw_data,_=readfile1(path)
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(raw_data)

    lda1 = LatentDirichletAllocation(max_iter=1000, learning_method='online', n_components=10)

    lda1.fit_transform(tf)

    # print("done in %0.3fs." % (time() - t0))
    tf_feature_names = tf_vectorizer.get_feature_names()
    topics = get_top_words(lda1, path1, tf_feature_names, 8, i=0, file1=res)
    print(topics)
