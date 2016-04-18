import matplotlib.pyplot as plt

from convRBM import CRBM
import getData as dataRead
import trainingObserver as observer
import buildModel

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
        'number_of_motifs': 20,
        'motif_length': 11,
        'learning_rate': 0.5,
        'doublestranded': True,
        'pooling_factor': 5,
        'epochs': 300,
        'cd_k': 5,
        'sparsity': 0.5,
        'batch_size': 200,
        'verbose': False,
        'cd_method': 'pcd',
        'momentum': 0.9
    }
]

train_test_ratio = 0.1
verificationSize = 500
########################################################


# read the data and split it
print "Reading the data..."
start = time.time()
seqReader = dataRead.SeqReader()
data = seqReader.readSequencesFromFile('../data/stemcells.fa')
# split
per = np.random.permutation(len(data))
itest = per[:int(len(data)*train_test_ratio)]
itrain = per[int(len(data)*train_test_ratio):]
veri = [random.randrange(0, len(itrain)) for i in range(verificationSize)]

# convert raw sequences to one hot matrices
start = time.time()
training_data = np.array([data[i] for i in itrain])
test_data = np.array([data[i] for i in itest])
verification_data = np.array([data[i] for i in veri])

print "Data successfully read in " + str((time.time()-start)) + " seconds."
print "Train set size: " + str(len(itrain))
print "Test set size: " + str(len(itest))

count = 0
# HERE, THE ACTUAL WORK IS DONE.
for hyper_params in allHyperParams:
    learner = buildModel.buildModelWithObservers(hyper_params, test_data, verification_data)
    learner.printHyperParams()

    try:  # We want to save the model, even when the script is terminated via Ctrl-C.
        start = time.time()
        learner.trainModel(training_data)
        dataRead.saveModel(learner,"test1.pkl")
        print "Training of " + str(training_data.shape[0]) + " performed in: " \
        		+ str(time.time()-start) + " seconds."

    except KeyboardInterrupt:
        print "You interrupted the program. It will save the model and exit."
        break

    finally:
        count += 1
