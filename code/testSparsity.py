
import matplotlib.pyplot as plt

from convRBM import CRBM
import getData as dataRead
import trainingObserver as observer

import numpy as np
import random
import time
from datetime import datetime
import os

from sklearn.metrics import roc_curve, auc
from scipy import interp

import theano
import theano.tensor as T


########################################################
# SET THE HYPER PARAMETERS
allHyperParams = [
    {
        'number_of_motifs': 100,
        'motif_length': 9,
        'learning_rate': 0.5,
        'doublestranded': False,
        'pooling_factor': 4,
        'epochs': 200,
        'cd_k': 5,
        'sparsity': 0.1,
        'batch_size': 20,
        'verbose': False,
        'cd_method': 'pcd',
        'momentum': 0.9
    },
    {
        'number_of_motifs': 100,
        'motif_length': 9,
        'learning_rate': 0.5,
        'doublestranded': False,
        'pooling_factor': 4,
        'epochs': 200,
        'cd_k': 5,
        'sparsity': 0.5,
        'batch_size': 20,
        'verbose': False,
        'cd_method': 'pcd',
        'momentum': 0.9
    },
    {
        'number_of_motifs': 100,
        'motif_length': 9,
        'learning_rate': 0.5,
        'doublestranded': False,
        'pooling_factor': 4,
        'epochs': 200,
        'cd_k': 5,
        'sparsity': 1,
        'batch_size': 20,
        'verbose': False,
        'cd_method': 'pcd',
        'momentum': 0.9
    },
    {
        'number_of_motifs': 100,
        'motif_length': 9,
        'learning_rate': 0.5,
        'doublestranded': False,
        'pooling_factor': 4,
        'epochs': 200,
        'cd_k': 5,
        'sparsity': 5,
        'batch_size': 20,
        'verbose': False,
        'cd_method': 'pcd',
        'momentum': 0.9
    },
    {
        'number_of_motifs': 100,
        'motif_length': 9,
        'learning_rate': 0.5,
        'doublestranded': False,
        'pooling_factor': 4,
        'epochs': 200,
        'cd_k': 5,
        'sparsity': 10,
        'batch_size': 20,
        'verbose': False,
        'cd_method': 'pcd',
        'momentum': 0.9
    },
    {
        'number_of_motifs': 100,
        'motif_length': 9,
        'learning_rate': 0.5,
        'doublestranded': False,
        'pooling_factor': 4,
        'epochs': 200,
        'cd_k': 5,
        'sparsity': 50,
        'batch_size': 20,
        'verbose': False,
        'cd_method': 'pcd',
        'momentum': 0.9
    }
]

testset_size = 100
batchSize = 100
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


def plotROC(scores, numPositives, sparsities):
    assert len(scores) == len(sparsities)
    fig = plt.figure(figsize=(14, 8))
    y_true = np.concatenate( (np.ones(numPositives), np.zeros(scores[0].shape[0] - numPositives)), axis=0 )

    for i in range(len(scores)):
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=scores[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='Sparsity lambda ' + str(sparsities[i]) + ' (AUC = %0.2f)' % roc_auc)

    plt.ylim([0.0, 1.05])
    plt.xlim([-0.05, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics for sparsity constraint')
    plt.legend(loc="lower right")
    fig.savefig('ROC_for_sparsity.png', dpi=400)

# get the data

print "Reading the data..."
start = time.time()
seqReader = dataRead.SeqReader()
data_stem = seqReader.readSequencesFromFile('../data/stemcells.fa')
data_fibro = seqReader.readSequencesFromFile('../data/fibroblast.fa')

trainingData_stem = np.array(data_stem[testset_size:])
testingData_stem = np.array(data_stem[:testset_size])
trainingData_fibro = np.array(data_fibro[testset_size:])
testingData_fibro = np.array(data_fibro[:testset_size])

allTestingData = np.concatenate( (testingData_stem, testingData_fibro), axis=0 )
numPositives = testingData_stem.shape[0]

print "Data successfully read in " + str((time.time()-start)) + " seconds."
print "Number of stemcell test samples: " + str(numPositives)
print "Number of fibroblast test samples: " + str(testingData_fibro.shape[0])

scores = list()
sparsities = list()
count = 1

for hyper_params in allHyperParams:
    # build model
    learner_stem = buildModelWithObservers(hyper_params, testingData_stem)
    learner_stem.printHyperParams()
    learner_fibro = buildModelWithObservers(hyper_params, testingData_fibro)

    # train model
    print "Train both models..."
    start = time.time()
    learner_stem.trainModel(trainingData_stem)
    learner_fibro.trainModel(trainingData_fibro)
    
    # evaluate free energy for testing data
    print "Get free energy for both models..."
    score_sc = getFreeEnergyPoints(learner_stem, allTestingData)
    score_fibro = getFreeEnergyPoints(learner_fibro, allTestingData)
    subtracted = score_fibro - score_sc
    
    scores.append(subtracted)
    sparsities.append(hyper_params['sparsity'])

    print "SPARSITY CONSTRAINT " + str(hyper_params['sparsity']) + " DONE (" + str(count) + " / " + str(len(allHyperParams)) + ")"
    count += 1

plotROC(scores, numPositives, sparsities)
