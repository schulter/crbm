
import matplotlib.pyplot as plt

from convRBM import CRBM
import getData as dataRead
import trainingObserver as observer

import numpy as np
import random
import time
from datetime import datetime
import os

plotAfterTraining = False
########################################################
# SET THE HYPER PARAMETERS
allHyperParams = [
    {
        'number_of_motifs': 100,
        'motif_length': 9,
        'learning_rate': 0.5,
        'doublestranded': False,
        'pooling_factor': 4,
        'epochs': 300,
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
        'epochs': 300,
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
        'epochs': 300,
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
        'epochs': 300,
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
        'epochs': 300,
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
        'epochs': 300,
        'cd_k': 5,
        'sparsity': 50,
        'batch_size': 20,
        'verbose': False,
        'cd_method': 'pcd',
        'momentum': 0.9
    }
]

train_test_ratio = 0.1
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


def saveModelAndPlot(model, name):
    # save trained model to file
    os.mkdir('../../training/' + name)
    file_name = "../../training/" + name + "/model.pkl"
    print "Saving model to " + str(file_name)
    model.saveModel(file_name)
    
    # do plotting



# read the data and split it
print "Reading the data..."
start = time.time()
seqReader = dataRead.SeqReader()
data_stem = seqReader.readSequencesFromFile('../data/stemcells.fa')
# split
per = np.random.permutation(len(data_stem))
itest = per[:int(len(data_stem)*train_test_ratio)]
itrain = per[int(len(data_stem)*train_test_ratio):]

# convert raw sequences to one hot matrices
trainingData_stem = np.array([data_stem[i] for i in itrain])
testingData_stem = np.array([data_stem[i] for i in itest])


data_fibro = seqReader.readSequencesFromFile('../data/fibroblast.fa')
# split
per = np.random.permutation(len(data_fibro))
itest = per[:int(len(data_fibro)*train_test_ratio)]
itrain = per[int(len(data_fibro)*train_test_ratio):]

# convert raw sequences to one hot matrices
trainingData_fibro = np.array([data_fibro[i] for i in itrain])
testingData_fibro = np.array([data_fibro[i] for i in itest])

print "Data successfully read in " + str((time.time()-start)) + " seconds."
print "Train set size: " + str(len(itrain))
print "Test set size: " + str(len(itest))

count = 0
# HERE, THE ACTUAL WORK IS DONE.
for hyper_params in allHyperParams:
    learner_stem = buildModelWithObservers(hyper_params, testingData_stem)
    learner_stem.printHyperParams()
    
    learner_fibro = buildModelWithObservers(hyper_params, testingData_fibro)
    learner_fibro.printHyperParams()

    try:  # We want to save the model, even when the script is terminated via Ctrl-C.
        start = time.time()
        learner_stem.trainModel(trainingData_stem)
        learner_fibro.trainModel(trainingData_fibro)
        print "Training of " + str(trainingData_stem.shape[0]) + " performed in: " + str(time.time()-start) + " seconds."

    except KeyboardInterrupt:
        print "You interrupted the program. It will save the model and exit."
        break

    finally:
        print "Save model number : " + str(count)
        saveModelAndPlot(learner_stem, "Training_stemcells_sparsity_" + str(hyper_params['sparsity']))
        saveModelAndPlot(learner_fibro, "Training_fibroblast_sparsity_" + str(hyper_params['sparsity']))
        count += 1
