
import matplotlib.pyplot as plt

from convRBM import CRBM
import getData as dataRead
import trainingObserver as observer
import markovModel
import numpy as np
import random
import time
from datetime import datetime
import os
import cPickle

from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn import svm

import theano
import theano.tensor as T


########################################################
# SET THE HYPER PARAMETERS
hyperParams={
        'number_of_motifs': 1,
        'motif_length': 9,
        'learning_rate': 0.5,
        'doublestranded': False,
        'pooling_factor': 4,
        'epochs': 200,
        'cd_k': 5,
        'sparsity': 0.5,
        'batch_size': 100,
        'verbose': False,
        'cd_method': 'pcd',
        'momentum': 0.9
    }

train_test_ratio = 0.1
batchSize = hyperParams['batch_size']
########################################################


def getObserver(model, title):
    for obs in model.observers:
        if title.lower() in obs.name.lower():
            return obs
    return None


def buildModelWithObservers(hyperParams, veri_data):
    model = CRBM(hyperParams=hyperParams)
    reconstruction_observer = observer.ReconstructionRateObserver(model,
                                                                  veri_data,
                                                                  "Reconstruction Rate Training Observer")
    model.addObserver(reconstruction_observer)
    return model


def saveModel(model, name):
    # save trained model to file
    os.mkdir('../../training/' + name)
    file_name = "../../training/" + name + "/model.pkl"
    print "Saving model to " + str(file_name)
    model.saveModel(file_name)


def getFreeEnergyFunction (model, data):
    D = T.tensor4('data')
    dataS = theano.shared(value=data, borrow=True, name='givenData')
    index = T.lscalar()
    energy = model.freeEnergyForData(D)
    return theano.function([index], energy, allow_input_downcast=True,
                           givens={D: dataS[index*batchSize:(index+1)*batchSize]},
                           name='freeDataEnergy'
                          )

def getFreeEnergyPoints(model, data):
    fun = getFreeEnergyFunction(model, data)
    iterations = data.shape[0] // batchSize

    M = np.zeros(data.shape[0])
    for batchIdx in xrange(iterations):
        #print "Setting from idx " + str(batchIdx*batchSize) + " to " + str((batchIdx+1)*batchSize)
        M[batchIdx*batchSize:(batchIdx+1)*batchSize] = fun(batchIdx)
    
    # to clean up the rest
    if data.shape[0] > iterations*batchSize:
        M[(batchIdx+1)*batchSize:] = fun(batchIdx+1)
    return M


def plotROC(scores, texts, labels):
    assert len(scores) == len(texts) #== len(labels)
    fig = plt.figure(figsize=(14, 8))

    for i in range(len(scores)):
        fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=scores[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=texts[i] + '(AUC = %0.2f)' % roc_auc)
    
    # plot the random line
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    # set general parameters for plotting and so on
    plt.ylim([0.0, 1.05])
    plt.xlim([-0.05, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics for sparsity constraint')
    plt.legend(loc="lower right")
    fig.savefig('ROC_for_motifnumber.png', dpi=400)

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

trainingData_stem = np.array([data_stem[i] for i in itrain_sc])
testingData_stem = np.array([data_stem[i] for i in itest_sc])
trainingData_fibro = np.array([data_fibro[i] for i in itrain_fibro])
testingData_fibro = np.array([data_fibro[i] for i in itest_fibro])

allTestingData = np.concatenate( (testingData_stem, testingData_fibro), axis=0 )
allTrainingData = np.concatenate( (trainingData_stem, trainingData_fibro), axis=0 )


print "Data successfully read in " + str((time.time()-start)) + " seconds."
print "Number of stemcell test samples: " + str(testingData_stem.shape[0])
print "Number of fibroblast test samples: " + str(testingData_fibro.shape[0])

scores = list()
texts = list()
count = 1

# add the markov model to scores
print "Training Markov Model on data..."
mm_sc = markovModel.MarkovModel()
mm_sc.trainModel(trainingData_stem)

mm_fib = markovModel.MarkovModel()
mm_fib.trainModel(trainingData_fibro)

scores_sc = mm_sc.evaluateSequences(allTestingData)
scores_fib = mm_fib.evaluateSequences(allTestingData)

scores.append(scores_sc - scores_fib)
texts.append('First Order Markov Model')

print "Training complete!"

# add the SVM to the plot
print "Training SVM on data..."
km_train = dataRead.computeKmerCounts(allTrainingData, 5)
km_test = dataRead.computeKmerCounts(allTestingData, 5)

labels_for_svm = np.concatenate( (np.ones(trainingData_stem.shape[0]), -np.ones(trainingData_fibro.shape[0])), axis=0 )

clf = svm.SVC(probability=True)
clf.fit(km_train, labels_for_svm)

scores.append(clf.predict_proba(km_test)[:,1])
texts.append('SVM with RBF kernel')
print "Training SVM complete"

allHyperParams=(1,5,10,50,100,500)
try:
    for nmot in allHyperParams:
        hyper_params=hyperParams
        #'number_of_motifs': 1,
        hyper_params['number_of_motifs']=nmot
        # build model
        learner_stem = buildModelWithObservers(hyper_params, testingData_stem)
        learner_stem.printHyperParams()
        learner_fibro = buildModelWithObservers(hyper_params, testingData_fibro)

        # train model
        print "Train cRBM for both datasets..."
        start = time.time()
        learner_stem.trainModel(trainingData_stem)
        learner_fibro.trainModel(trainingData_fibro)
        
        # evaluate free energy for testing data
        print "Get free energy for both models..."
        score_sc = getFreeEnergyPoints(learner_stem, allTestingData)
        score_fibro = getFreeEnergyPoints(learner_fibro, allTestingData)
        subtracted = score_fibro - score_sc

        scores.append(subtracted)
        texts.append('cRBM with ' + str(hyper_params['number_of_motifs']) + ' motifs')

        print "MOTIF NUMBER " + str(hyper_params['number_of_motifs']) + " DONE (" + str(count) + " / " + str(len(allHyperParams)) + ")"
        count += 1
except KeyboardInterrupt:
    print "Ending the program now. But let me first save the scores calculated!"

finally:
    labels = np.concatenate( (np.ones(testingData_stem.shape[0]), np.zeros(testingData_fibro.shape[0])), axis=0 )
    with open('scores_motif_number_test.pkl', 'w') as f:
        cPickle.dump( (scores, texts, labels), f)
    plotROC(scores, texts, labels)

