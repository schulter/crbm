from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import getData as dataRead

import numpy as np
import os



train_test_ratio = 0.1

print "Reading the data..."
seqReader = dataRead.SeqReader()
data_stem = seqReader.readSequencesFromFile('../data/stemcells.fa')
# split
per = np.random.permutation(len(data_stem))
itest = per[:int(len(data_stem)*train_test_ratio)]
itrain = per[int(len(data_stem)*train_test_ratio):]

# convert raw sequences to one hot matrices
training_stem = np.array([data_stem[i] for i in itrain])
test_stem = np.array([data_stem[i] for i in itest])
km_stem_training=dataRead.computeKmerCounts(training_stem,5)
km_stem_test=dataRead.computeKmerCounts(test_stem,5)

data_fibro = seqReader.readSequencesFromFile('../data/fibroblast.fa')
# split
per = np.random.permutation(len(data_fibro))
itest = per[:int(len(data_fibro)*train_test_ratio)]
itrain = per[int(len(data_fibro)*train_test_ratio):]

training_fibro = np.array([data_fibro[i] for i in itrain])
test_fibro = np.array([data_fibro[i] for i in itest])
km_fibro_training=dataRead.computeKmerCounts(training_fibro,5)
km_fibro_test=dataRead.computeKmerCounts(test_fibro,5)

training_input=np.concatenate((km_stem_training,km_fibro_training),axis=0)
training_labels=np.ones((training_input.shape[0]))
training_labels[:km_stem_training.shape[0]]=-1

test_input=np.concatenate((km_stem_test,km_fibro_test),axis=0)
test_labels=np.ones((test_input.shape[0]))
test_labels[:km_stem_test.shape[0]]=-1

print "Finished reading and converting. Start classifying"
# train an SVM
clf = svm.SVC()
clf.fit(training_input, training_labels)

# predict on held out test set
#clf.predict(test_input)
test_predicted=clf.decision_function(test_input)

# inspect the performance
print roc_auc_score(test_labels, test_predicted)
roc_curve(test_labels,test_predicted)
