import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

from convRBM import CRBM
import getData as dataRead
import trainingObserver as observer

import numpy as np
import random
import time
from datetime import datetime
import cPickle

# read the data
seqReader = dataRead.FASTAReader()
allSeqs = seqReader.readSequencesFromFile('../data/wgEncodeAwgDnaseUwAg10803UniPk.fa')

data = [allSeqs[random.randrange(0,len(allSeqs))] for i in range(20000)]
#data = allSeqs
train_set, test_set = train_test_split(data, test_size=0.01)
print "Training set size: " + str(len(train_set))
print "Test set size: " + str(len(test_set))

start = time.time()
trainingData = np.array([dataRead.getOneHotMatrixFromSeq(t) for t in train_set])
testingData = np.array([dataRead.getOneHotMatrixFromSeq(t) for t in test_set])
print "Conversion of test set in (in ms): " + str((time.time()-start)*1000)

learner = CRBM(7, 10, 0.0001, 2)

# add the observers for free energy (test and train)
free_energy_observer = observer.FreeEnergyObserver(learner, testingData, "Free Energy Testing Observer")
learner.addObserver(free_energy_observer)
free_energy_train_observer = observer.FreeEnergyObserver(learner, trainingData, "Free Energy Training Observer")
#learner.addObserver(free_energy_train_observer)

# add the observers for reconstruction error (test and train)
reconstruction_observer = observer.ReconstructionErrorObserver(learner, testingData, "Reconstruction Error Testing Observer")
learner.addObserver(reconstruction_observer)
reconstruction_observer_train = observer.ReconstructionErrorObserver(learner, trainingData, "Reconstruction Error Training Observer")
#learner.addObserver(reconstruction_observer_train)

# add the observer of the motifs during training (to look at them afterwards)
motif_observer = observer.MotifObserver(learner, trainingData)
learner.addObserver(motif_observer)

print "Data mat shape: " + str(trainingData.shape)

# set PARAMETERS FOR EPOCHS AND CD
epochs = 2
cd_k = 1

start = time.time()
learner.trainMinibatch(trainingData, epochs, 50, cd_k)
print "Training of " + str(trainingData.shape[0]) + " performed in: " + str(time.time()-start) + " seconds."

# save trained model to file
file_name = datetime.now().strftime("../../models/trainedModel_%Y_%m_%d_%H_%M.pkl")
print "Saving model to " + str(file_name)
learner.saveModel(file_name)

plt.subplot(2,1,1)
plt.ylabel('Free energy function')
plt.xlabel('Number Of Epoch')
plt.title('Free Energy')
plt.plot(free_energy_observer.scores)
plt.plot(free_energy_train_observer.scores)

plt.subplot(2,1,2)
plt.ylabel('Reconstruction error on dataset')
plt.xlabel('Number Of Epoch')
plt.title('Reconstruction Error')
plt.plot(reconstruction_observer.scores)
plt.plot(reconstruction_observer_train.scores)

# save plot to file
file_name_plot = "training_" + str(epochs) + "epo_" + str(learner.motifLength) + "kmers_" + str(learner.numMotifs) + "motifs_CD"+str(cd_k)+".png"
plt.savefig(file_name_plot)
