# Imports
import sys
import os
import random
import time
import numpy as np
import theano
import theano.tensor as T
np.set_printoptions(precision=2, suppress=True)

# the underlying convRBM implementation
sys.path.append(os.path.abspath('../code'))
from convRBM import CRBM
from convRBM_naive import NaiveCRBM
import getData as dataRead

# biopython stuff
#import Bio.SeqIO as sio
#import Bio.motifs.matrix as mat
#from Bio.Alphabet import IUPAC
#from Bio.Seq import Seq
#from Bio import motifs
#import math

# some test specific parameters
test_failures = 0
tests_total = 0
desired_precision = 0.0001
number_of_seqs = 200

# read the data
seqReader = dataRead.SeqReader()
print "Loading dataset ..."
data = seqReader.readOneHotFromFile('../data/seq.onehot.gz')
#per=np.random.permutation(len(data))
per = range(0,len(data))
isubset=per[:number_of_seqs]
data=np.array([data[i] for i in isubset])
print "Data successfully read with shape: " + str(data.shape)

# finalize setup params for the tests
test_names=["Comparision of training of real sequences"]
test_params=[
{'number_of_motifs':2,
                'motif_length':11,
                'learning_rate':1,
                'pooling_factor':1,
                'epochs':10,
                'cd_k':1,
                'batch_size':min(data.shape[0], 200),
								'sparsity':0,
								'rho':0.001,
								'verbose':False,
								'momentum':0,
                'doublestranded':False
}]

# execute for every test

for name, param in zip(test_names, test_params):
    print "----------------------"
    print name
    print "----------------------"

    gpuModel = CRBM(param)
    naiveModel = NaiveCRBM(param)

    naiveModel.setMotifs(gpuModel.motifs.get_value())
    naiveModel.setBiases(gpuModel.bias.get_value())
    naiveModel.setC(gpuModel.c.get_value())

    print "Start training naive model with params:"
    start = time.time()
    gpuModel.printHyperParams()
    naiveModel.trainModel(data)
    print "Training the naive model took: " + str(time.time()-start) + " seconds"
    print "Done training naive Model. Start with GPU model..."
    gpuModel.trainModel(data)

    diff_motif = np.mean(np.abs(gpuModel.motifs.get_value() - naiveModel.motifs))
    diff_bias = np.mean(np.abs(gpuModel.bias.get_value() - naiveModel.bias))
    diff_c = np.mean(np.abs(gpuModel.c.get_value() - naiveModel.c))
    
    if diff_motif <= desired_precision:
        print "Training Test (motifs): PASSED [Error: " + str(diff_motif) + "]"
        tests_total += 1
    else:
        print "Training Test (motifs): FAILED [Error: " + str(diff_motif) + "]"
        tests_total += 1
        test_failures += 1
    
    if diff_bias <= desired_precision:
        print "Training Test (bias): PASSED! [Error: " + str(diff_bias) + "]"
        tests_total += 1
    else:
        print "Training Test (bias): FAILED [Error: " + str(diff_bias) + "]"
        tests_total += 1
        test_failures += 1
    
    if diff_c <= desired_precision:
        print "Training Test (c): PASSED [Error: " + str(diff_c) + "]"
        tests_total += 1
    else:
        print "Training Test (c): FAILED [Error: " + str(diff_c) + "]"
        tests_total += 1
        test_failures += 1
    
