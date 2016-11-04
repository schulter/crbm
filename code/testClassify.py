


import matplotlib.pyplot as plt

from convRBM import CRBM
import getData as dataRead

import markovModel
import numpy as np
import random
import time
from datetime import datetime
import os
import cPickle

from scipy import interp
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA as pca

import theano
import theano.tensor as T


########################################################
# SET THE HYPER PARAMETERS
hyperParams={
        'number_of_motifs': 42,
        'motif_length': 15,
        'input_dims':4,
        'momentum':0.95,
        'learning_rate': .5,
        'doublestranded': True,
        'pooling_factor': 4,
        'epochs': 300,
        'cd_k': 5,
        'sparsity': 0.5,
        'batch_size': 100,
    }

np.set_printoptions(precision=4)
train_test_ratio = 0.1
batchSize = hyperParams['batch_size']
########################################################

# get the data

print "Reading the data..."
start = time.time()
seqReader = dataRead.SeqReader()
data_stem = seqReader.readSequencesFromFile('../data/stemcells.fa')
data_fibro = seqReader.readSequencesFromFile('../data/fibroblast.fa')[:len(data_stem)]

per_sc = np.random.permutation(len(data_stem))
itest_sc = per_sc[:int(len(data_stem)*train_test_ratio)]
itrain_sc = per_sc[int(len(data_stem)*train_test_ratio):]

per_fibro = np.random.permutation(len(data_fibro))
itest_fibro = per_fibro[:int(len(data_fibro)*train_test_ratio)]
itrain_fibro = per_fibro[int(len(data_fibro)*train_test_ratio):]

training_stem = np.array([data_stem[i] for i in itrain_sc])
test_stem = np.array([data_stem[i] for i in itest_sc])
training_fibro = np.array([data_fibro[i] for i in itrain_fibro])
test_fibro = np.array([data_fibro[i] for i in itest_fibro])

allTest = np.concatenate( (test_stem, test_fibro), axis=0 )
allTraining = np.concatenate( (training_stem, training_fibro), axis=0 )
nseq=int((allTest.shape[3]-hyperParams['motif_length'] + 1)/hyperParams['pooling_factor'])*\
        		hyperParams['pooling_factor']+ hyperParams['motif_length'] -1
allTest=allTest[:,:,:,:nseq]
allTraining=allTraining[:,:,:,:nseq]


crbm_stem = CRBM(hyperParams=hyperParams)
crbm_fibro = CRBM(hyperParams=hyperParams)

# train model
print "Train cRBM for both datasets..."
start = time.time()
crbm_stem.trainModel(training_stem,test_stem)
crbm_fibro.trainModel(training_fibro,test_fibro)

# evaluate free energy for testing data
print "Get free energy for both models..."
score_sc = crbm_stem.freeEnergy(allTest)
score_fibro = crbm_fibro.freeEnergy(allTest)
score = score_fibro - score_sc

labels = np.concatenate( (np.ones(test_stem.shape[0]), np.zeros(test_fibro.shape[0])), axis=0 )
auc=metrics.roc_auc_score(labels,score)
prc=metrics.average_precision_score(labels,score)
print("auc: "+str(auc))
print("prc: "+str(prc))
crbm_stem.printHyperParams()
