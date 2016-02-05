
from convRBM import CRBM
import getData as dataRead

import numpy as np
import random
import time
from datetime import datetime
import theano

import os

seqReader = dataRead.SeqReader()
alls=seqReader.readOneHotFromFile('../data/seq.onehot.gz')

data = [alls[random.randrange(0,len(alls))] for i in range(2000)]

per=np.random.permutation(len(data))
itest=per[:int(len(data)*0.01)]
itrain=per[int(len(data)*0.01):]
start = time.time()
train_data = np.array([data[i] for i in itrain])
test_data = np.array([data[i] for i in itest])
print "Conversion of test set in (in ms): " + str((time.time()-start)*1000)

epochs = 900
cd_k = 5
learning_rate = 0.00001
doublestranded=True
motif_length = 11
number_of_motifs = 2
batch_size = 100
pooling_factor = 5
sparsity=0.001 # regularization parameter
rho=0.001 # threshold for motif hit frequency
train_test_ratio = 0.01
USE_WHOLE_DATA = True

hyper_params = {'number_of_motifs':number_of_motifs,
				'motif_length':motif_length,
				'learning_rate':learning_rate,
				'doublestranded':doublestranded,
				'pooling_factor':pooling_factor,
				'epochs':epochs,
				'cd_k':cd_k,
				'sparsity':sparsity,
				'rho':rho,
				'batch_size':batch_size
							}

learner = CRBM(hyperParams=hyper_params)
#learner.trainModel(trainingData)


data=theano.tensor.tensor4(name='data')
out=learner.computeHgivenV(data)
#compile computeHgivenV
computeHgivenV=theano.function([data],out,allow_input_downcast=True)

hidden=theano.tensor.tensor4(name='hidden')
out=learner.collectUpdateStatistics(hidden,data)
#compile collectUpdateStatistics
collectUpdateStatistics=theano.function([hidden,data],out,allow_input_downcast=True)

data=theano.tensor.tensor4(name='data')
out=learner.computeVgivenH(data)
#compile computeHgivenV
computeVgivenH=theano.function([data],out,allow_input_downcast=True)

ph,h=computeHgivenV(train_data)
evh_data,eh_data,ev_data=collectUpdateStatistics(h,train_data)
for i in range(100):
    pv,v=computeVgivenH(h)
    ph,h=computeHgivenV(v)

evh_model,eh_model,ev_model=collectUpdateStatistics(h,v)
abs(evh_data-evh_model).sum()
learner.motifs.set_value(learner.motifs.get_value()+0.0001*(evh_data-evh_model))
learner.bias.set_value(learner.bias.get_value()+0.0001*(eh_data-eh_model))
learner.c.set_value(learner.c.get_value()+0.0001*(ev_data-ev_model))
learner.motifs.get_value()[0,0]
evh_data[0,0]
evh_model[0,0]

pv,v=computeVgivenH(h)
ph,h=computeHgivenV(v)
pv,v=computeVgivenH(h)
ph,h=computeHgivenV(v)
evh,eh,ev=collectUpdateStatistics(ph,test_data)

import theano.tensor.nnet.conv as conv

D=theano.tensor.tensor4('data')
M=theano.tensor.tensor4('motif')
out=conv.conv2d(D,M,border_mode='full')
conv=theano.function([D,M],out, allow_input_downcast=True)
mot=learner.motifs.get_value()
mot=np.swapaxes(mot,1,0)
vr=conv(h,mot)

