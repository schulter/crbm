import matplotlib.pyplot as plt

from convRBM import CRBM
import getData as dataRead
import trainingObserver as observer

import numpy as np
import random
import time
from datetime import datetime
import cPickle
import theano
import os

plotAfterTraining = False
########################################################
# SET THE HYPER PARAMETERS
allHyperParams = [
{'number_of_motifs':100,
'motif_length':11,
'learning_rate':0.5,
'doublestranded':False,
'pooling_factor':5,
'epochs':100,
'cd_k':5,
'sparsity':1.01,
'rho':0.05,
'batch_size':100,
'verbose':False,
'cd_method':'pcd',
'momentum':0.9
},
{'number_of_motifs':150,
'motif_length':11,
'learning_rate':0.5,
'doublestranded':False,
'pooling_factor':5,
'epochs':100,
'cd_k':5,
'sparsity':1.01,
'rho':0.05,
'batch_size':100,
'verbose':False,
'cd_method':'pcd',
'momentum':0.9
},
{'number_of_motifs':300,
'motif_length':11,
'learning_rate':0.5,
'doublestranded':False,
'pooling_factor':5,
'epochs':100,
'cd_k':5,
'sparsity':1.01,
'rho':0.05,
'batch_size':50,
'verbose':False,
'cd_method':'pcd',
'momentum':0.9
},
{'number_of_motifs':500,
'motif_length':11,
'learning_rate':0.5,
'doublestranded':False,
'pooling_factor':5,
'epochs':100,
'cd_k':5,
'sparsity':1.01,
'rho':0.05,
'batch_size':20,
'verbose':False,
'cd_method':'pcd',
'momentum':0.9
}]

train_test_ratio = 0.1
verificationSize = 500
########################################################

def getObserver(model, title):
    for obs in model.observers:
        if title.lower() in obs.name.lower():
            return obs
    return None


def buildModelWithObservers (hyper_params, testingData, verificationData):
    model = CRBM(hyperParams=hyper_params)

    # add the observers for free energy (test and train)
    free_energy_observer = observer.FreeEnergyObserver(model, testingData, "Free Energy Testing Observer")
    model.addObserver(free_energy_observer)
    free_energy_train_observer = observer.FreeEnergyObserver(model, verificationData, "Free Energy Training Observer")
    model.addObserver(free_energy_train_observer)

    # add the observers for reconstruction error (test and train)
    reconstruction_observer = observer.ReconstructionRateObserver(model, testingData, "Reconstruction Rate Testing Observer")
    model.addObserver(reconstruction_observer)
    reconstruction_observer_train = observer.ReconstructionRateObserver(model, verificationData, "Reconstruction Rate Training Observer")
    model.addObserver(reconstruction_observer_train)

    # add the observer of the motifs during training (to look at them afterwards)
    param_observer = observer.ParameterObserver(model, None)
    model.addObserver(param_observer)

    # add the motif hit scanner
    motif_hit_observer = observer.MotifHitObserver(model, testingData)
    model.addObserver(motif_hit_observer)

    # add IC observers
    icObserver = observer.InformationContentObserver(model)
    model.addObserver(icObserver)

    medianICObserver = observer.MedianICObserver(model)
    model.addObserver(medianICObserver)

    return model


def saveModelAndPlot(model):
    # save trained model to file
    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir('../../training/' + date_string)
    file_name = "../../training/" + date_string + "/model.pkl"
    print "Saving model to " + str(file_name)
    model.saveModel(file_name)

    if plotAfterTraining:
        # plot
        plt.subplot(2,1,1)
        plt.ylabel('Free energy function')
        title = "%f lr %d kmers %d numOfMotifs %d cd_k" % (model.hyper_params['learning_rate'],
                                                           model.hyper_params['motif_length'],
                                                           model.hyper_params['number_of_motifs'],
                                                           model.hyper_params['cd_k'])
        plt.title(title)

        plt.plot(getObserver(model, "free energy testing").scores)
        plt.plot(getObserver(model, "free energy training").scores)

        plt.subplot(2,1,2)
        plt.ylabel('Reconstruction rate on dataset')
        plt.xlabel('Number Of Epoch')

        plt.plot(getObserver(model, "reconstruction rate training").scores)
        plt.plot(getObserver(model, "reconstruction rate testing").scores)

        # save plot to file
        file_name_plot = "../../training/" + date_string + "/errorPlot.png"
        plt.savefig(file_name_plot)



# read the data and split it
print "Reading the data..."
start = time.time()
seqReader = dataRead.SeqReader()
data = seqReader.readOneHotFromFile('../data/seq.onehot.gz')
# split
per=np.random.permutation(len(data))
itest=per[:int(len(data)*train_test_ratio)]
itrain=per[int(len(data)*train_test_ratio):]
veri = [random.randrange(0, len(itrain)) for i in range(verificationSize)]

# convert raw sequences to one hot matrices
start = time.time()
trainingData = np.array([data[i] for i in itrain])
testingData = np.array([data[i] for i in itest])
verificationData = np.array([data[i] for i in veri])

print "Data successfully read in " + str((time.time()-start)) + " seconds."
print "Train set size: " + str(len(itrain))
print "Test set size: " + str(len(itest))

count = 0
# HERE, THE ACTUAL WORK IS DONE.
for hyper_params in allHyperParams:
    learner = buildModelWithObservers(hyper_params, testingData, verificationData)
    learner.printHyperParams()

    try:# We want to save the model, even when the script is terminated via Ctrl-C.
        start = time.time()
        learner.trainModel(trainingData)
        print "Training of " + str(trainingData.shape[0]) + " performed in: " + str(time.time()-start) + " seconds."

    except KeyboardInterrupt:
        print "You interrupted the program. It will save the model and exit."

    finally:
        print "Save model number : " + str(count)
        saveModelAndPlot(learner)
        count += 1
