#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from sklearn.cross_validation import train_test_split

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

########################################################
# SET THE HYPER PARAMETERS
epochs = 300
cd_k = 5
learning_rate = 0.5
doublestranded = False
motif_length = 11
number_of_motifs = 100
batch_size = 100
pooling_factor = 5
sparsity = 1.01 # regularization parameter
rho=0.005 # threshold for motif hit frequency

train_test_ratio = 0.1

USE_WHOLE_DATA = True
validationSize = 500
########################################################



# read the data and split it
seqReader = dataRead.SeqReader()
allSeqs = seqReader.readOneHotFromFile('../data/seq.onehot.gz')

if USE_WHOLE_DATA:
	data = allSeqs
else:
	data = [allSeqs[random.randrange(0,len(allSeqs))] for i in range(2000)]

per=np.random.permutation(len(data))
itest=per[:int(len(data)*train_test_ratio)]
itrain=per[int(len(data)*train_test_ratio):]

print "Test set size: " + str(len(itest))
print "Train set size: " + str(len(itrain))

val = [random.randrange(0, len(itrain)) for i in range(validationSize)]

# convert raw sequences to one hot matrices
start = time.time()
trainingData = np.array([data[i] for i in itrain])
testingData = np.array([data[i] for i in itest])
validationData = np.array([data[i] for i in val])

print "Conversion of test set in (in ms): " + str((time.time()-start)*1000)

# construct the model

hyper_params = {'number_of_motifs':number_of_motifs,
				'motif_length':motif_length,
				'learning_rate':learning_rate,
				'doublestranded':doublestranded,
				'biases_to_zero':False,
				'pooling_factor':pooling_factor,
				'epochs':epochs,
				'cd_k':cd_k,
				'sparsity':sparsity,
				'rho':rho,
				'batch_size':batch_size,
				'verbose':False,
				'cd_method':'pcd',
				'momentum':0.9 # use 0.0 to disable momentum
}
learner = CRBM(hyperParams=hyper_params)

# add the observers for free energy (test and train)
free_energy_observer = observer.FreeEnergyObserver(learner, testingData, "Free Energy Testing Observer")
learner.addObserver(free_energy_observer)
#free_energy_train_observer = observer.FreeEnergyObserver(learner, trainingData, "Free Energy Training Observer")
#learner.addObserver(free_energy_train_observer)

# add the observers for reconstruction error (test and train)
reconstruction_observer = observer.ReconstructionRateObserver(learner, testingData, "Reconstruction Rate Testing Observer")
learner.addObserver(reconstruction_observer)
#reconstruction_observer_train = observer.ReconstructionRateObserver(learner, trainingData, "Reconstruction Rate Training Observer")
#learner.addObserver(reconstruction_observer_train)

# add the observer of the motifs during training (to look at them afterwards)
param_observer = observer.ParameterObserver(learner, None)
learner.addObserver(param_observer)

# add the motif hit scanner
motif_hit_observer = observer.MotifHitObserver(learner, testingData)
learner.addObserver(motif_hit_observer)
print "Data mat shape: " + str(trainingData.shape)

# add IC observers
icObserver = observer.InformationContentObserver(learner)
learner.addObserver(icObserver)

medianICObserver = observer.MedianICObserver(learner, trainingData)
learner.addObserver(medianICObserver)

# HERE, THE ACTUAL WORK IS DONE.
# We want to save the model, even when the script is terminated via Ctrl-C.


def saveModelAndPlot():
	# save trained model to file
	date_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
	os.mkdir('../../training/' + date_string)
	file_name = "../../training/" + date_string + "/model.pkl"
	print "Saving model to " + str(file_name)
	learner.saveModel(file_name)
	
	# plot
	plt.subplot(2,1,1)
	plt.ylabel('Free energy function')
	plt.title(str(hyper_params['learning_rate']) + " lr " + str(motif_length) + " kmers " + str(number_of_motifs) + " motifs_CD "+str(cd_k)+".png")

	plt.plot(free_energy_observer.scores)
	plt.plot(free_energy_train_observer.scores)

	plt.subplot(2,1,2)
	plt.ylabel('Reconstruction rate on dataset')
	plt.xlabel('Number Of Epoch')
	plt.plot(reconstruction_observer.scores)
	plt.plot(reconstruction_observer_train.scores)

	# save plot to file
	file_name_plot = "../../training/" + date_string + "/errorPlot.png"
	plt.savefig(file_name_plot)

learner.printHyperParams()
try:
	# perform training
	start = time.time()
	learner.trainModel(trainingData)
	print "Training of " + str(trainingData.shape[0]) + " performed in: " + str(time.time()-start) + " seconds."

except KeyboardInterrupt:
	saveModelAndPlot()
	print "You interrupted the program. It will save the model and exit."
	raise


saveModelAndPlot()
