
import matplotlib.pyplot as plt

from convRBM import CRBM
import getData as dataRead
import trainingObserver as observer
import buildModel
import plotting
import freeenergy

import markovModel
import numpy as np
import random
import time
from datetime import datetime
import os
import cPickle

from scipy import interp
from sklearn import svm

import theano
import theano.tensor as T


########################################################
# SET THE HYPER PARAMETERS
hyperParams={
        'number_of_motifs': 10,
        'motif_length': 7,
        'learning_rate': .5,
        'doublestranded': True,
        'pooling_factor': 4,
        'epochs': 500,
        'cd_k': 5,
        'sparsity': 0.5,
        'batch_size': 200,
        'verbose': False,
        'cd_method': 'pcd',
        'momentum': 0.9
    }

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


print "Data successfully read in " + str((time.time()-start)) + " seconds."
print "Number of stemcell test samples: " + str(test_stem.shape[0])
print "Number of fibroblast test samples: " + str(test_fibro.shape[0])

scores = list()
texts = list()
count = 1

# add the markov model to scores
print "Training Markov Model on data..."
mm_sc = markovModel.MarkovModel()
mm_sc.trainModel(training_stem)

mm_fib = markovModel.MarkovModel()
mm_fib.trainModel(training_fibro)

scores_sc = mm_sc.evaluateSequences(allTest)
scores_fib = mm_fib.evaluateSequences(allTest)

scores.append(scores_sc - scores_fib)
texts.append('First Order Markov Model')

print "Training complete!"

# add the SVM to the plot
print "Training SVM on data..."
km_train = dataRead.computeKmerCounts(allTraining, 5)
km_test = dataRead.computeKmerCounts(allTest, 5)

labels_for_svm = np.concatenate( (np.ones(training_stem.shape[0]), -np.ones(training_fibro.shape[0])), axis=0 )

clf = svm.SVC(probability=False)
clf.fit(km_train, labels_for_svm)
#
scores.append(clf.decision_function(km_test))
texts.append('SVM with RBF kernel')
print "Training SVM complete"

allHyperParams=(1,5,10,20,50)
try:
    for nmot in allHyperParams:
        hyper_params=hyperParams
        #'number_of_motifs': 1,
        hyper_params['number_of_motifs']=nmot
        # build model
        learner_stem = buildModel.buildModelWithObservers(hyper_params, test_stem,training_fibro)
        learner_stem.printHyperParams()
        learner_fibro = buildModel.buildModelWithObservers(hyper_params, test_fibro,training_fibro)

        # train model
        print "Train cRBM for both datasets..."
        start = time.time()
        learner_stem.trainModel(training_stem)
        learner_fibro.trainModel(training_fibro)
        
        # evaluate free energy for testing data
        print "Get free energy for both models..."
        score_sc = freeenergy.getFreeEnergyPoints(learner_stem, allTest)
        score_fibro = freeenergy.getFreeEnergyPoints(learner_fibro, allTest)
        subtracted = score_fibro - score_sc

        scores.append(subtracted)
        texts.append('cRBM with ' + str(hyper_params['number_of_motifs']) + ' motifs')

        print "MOTIF NUMBER " + str(hyper_params['number_of_motifs']) + " DONE (" + str(count) + " / " + str(len(allHyperParams)) + ")"
        count += 1
except KeyboardInterrupt:
    print "Ending the program now. But let me first save the scores calculated!"

finally:
    labels = np.concatenate( (np.ones(test_stem.shape[0]), np.zeros(test_fibro.shape[0])), axis=0 )
    with open('scores_motif_number_test.pkl', 'w') as f:
        cPickle.dump( (scores, texts, labels), f)
    plotting.plotROC(scores, texts, labels,'roc_nummotifs.png')

