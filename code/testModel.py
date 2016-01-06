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
train_set, test_set = train_test_split(data, test_size=0.1)
print "Training set size: " + str(len(train_set))
print "Test set size: " + str(len(test_set))

start = time.time()
trainingData = np.array([dataRead.getMatrixFromSeq(t) for t in train_set])
testingData = np.array([dataRead.getMatrixFromSeq(t) for t in test_set])
print "Conversion of test set in (in ms): " + str((time.time()-start)*1000)

learner = CRBM(9, 20, 0.001, 2)

# add the observers for free energy (test and train)
free_energy_observer = observer.FreeEnergyObserver(learner, testingData)
learner.addObserver(free_energy_observer)
free_energy_train_observer = observer.FreeEnergyObserver(learner, trainingData)
learner.addObserver(free_energy_train_observer)

# add the observers for reconstruction error (test and train)
reconstruction_observer = observer.ReconstructionErrorObserver(learner, testingData)
learner.addObserver(reconstruction_observer)
reconstruction_observer_train = observer.ReconstructionErrorObserver(learner, trainingData)
learner.addObserver(reconstruction_observer_train)

print "Data mat shape: " + str(trainingData.shape)
start = time.time()
learner.trainMinibatch(trainingData, testingData, 10, 50, 5)
print "Training of " + str(trainingData.shape[0]) + " performed in: " + str(time.time()-start) + " seconds."

# save trained model to file
file_name = datetime.now().strftime("trainedModel_%Y_%m_%d_%H_%M.pkl")
with open('models/' + file_name, 'w') as f:
	cPickle.dump(learner, f)

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

plt.savefig('longRun_1000epo_9kmers_cd5.png')
