from sklearn import svm

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
trainingData_stem = np.array([data_stem[i] for i in itrain])
testingData_stem = np.array([data_stem[i] for i in itest])
km_stem_training=dataRead.computeKmerCounts(trainingData_stem,5)
km_stem_test=dataRead.computeKmerCounts(trainingData_stem,5)

data_fibro = seqReader.readSequencesFromFile('../data/fibroblast.fa')
# split
per = np.random.permutation(len(data_fibro))
itest = per[:int(len(data_fibro)*train_test_ratio)]
itrain = per[int(len(data_fibro)*train_test_ratio):]

# convert raw sequences to one hot matrices
trainingData_fibro = np.array([data_fibro[i] for i in itrain])
testingData_fibro = np.array([data_fibro[i] for i in itest])
km_fibro_training=dataRead.computeKmerCounts(trainingData_fibro,5)
km_fibro_test=dataRead.computeKmerCounts(trainingData_fibro,5)

training_input=np.concatenate((km_stem_training,km_fibro_training),axis=1)
training_labels=np.ones((1,training_input.shape[1]))
training_labels[0,:km_stem_training.shape[0]]=-1

test_input=np.concatenate((km_stem_test,km_fibro_test),axis=1)
test_labels=np.ones((1,test_input.shape[1]))
test_labels[0,:km_stem_test.shape[0]]=-1
#np.concatenate((-np.ones((1,training_input.shape[1])),\
#np.ones((1,training_input.shape[1]))))

clf = svm.SVC(probability=True)
clf.fit(training_input.T, training_labels[0,:])

clf.predict(test_input)
clf.predict_proba(test_input)
