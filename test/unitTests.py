# Imports
import sys
import os
import random
import time
import numpy as np
np.set_printoptions(precision=2, suppress=True)

# the underlying convRBM implementation
sys.path.append(os.path.abspath('../code'))
from convRBM import CRBM
from convRBM_naive import NaiveCRBM
import getData as dataRead

# biopython stuff
#import Bio.SeqIO as sio
#import Bio.motifs.matrix as mat
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
#from Bio import motifs
import math


# CONSTRUCT THE TOY DATA AND TEST ALL RELEVANT METHODS ON IT
# ----------------------------------------------------------
kernel1 = np.tile(np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]]), [1,1,1])
kernel1_ = np.tile(np.flipud(np.fliplr(kernel1[0])),[1,1,1])
kernel2 = np.tile(np.array([[0,0,0],[0,0,0],[1,1,1],[0,0,0]]), [1,1,1])
kernel2_ = np.tile(np.flipud(np.fliplr(kernel2[0])), [1,1,1])
kernel3 = np.random.rand(1,4,3)
kernel3_ = np.tile(np.flipud(np.fliplr(kernel3[0])), [1,1,1])
kernel = np.array([kernel1, kernel1_])#, kernel2, kernel2_])#, kernel3, kernel3_])

# initialize the data
randSeq1 = dataRead.getOneHotSeq(Seq("ACGTGGGG", IUPAC.unambiguous_dna))
randSeq2 = dataRead.getOneHotSeq(Seq("ACGTACGT", IUPAC.unambiguous_dna))
data = np.array([randSeq1], dtype=np.float32)


#initialize the learner and set custom kernels
hyper_params = {'number_of_motifs':1,
                'motif_length':3,
                'learning_rate':0.1,
                'pooling_factor':1,
                'epochs':100,
                'cd_k':1,
                'batch_size':1
}
naiveModel = NaiveCRBM(motifLength=hyper_params['motif_length'],
                       numMotifs=hyper_params['number_of_motifs'],
                       learningRate=hyper_params['learning_rate'],
                       poolingFactor=hyper_params['pooling_factor'])

gpuModel = CRBM(hyper_params)
gpuModel.setToZero = True
# set parameters
naiveModel.setCustomKernels(kernel)
gpuModel.setCustomKernels(kernel)
gpuModel.batchSize = 1
gpuModel.printHyperParams()
gpuModel.debug = False
naiveModel.debug = False

import theano
import theano.tensor as T
import theano.tensor.nnet.conv as conv


test_failures = 0
tests_total = 0
desired_precision = 0.01
iterations = 1000


# create theano functions
# forward
print "Compiliing theano functions..."
D = T.tensor4('data')
[P_H, H] = gpuModel.computeHgivenV(D)
forward = theano.function([D], [P_H,H], allow_input_downcast=True)

# backward
H = T.tensor4('Hidden')
[P_V, V] = gpuModel.computeVgivenH(H)
backward = theano.function([H], [P_V,V], allow_input_downcast=True)

# gradient
H = T.tensor4('Hidden Probabilities')
D = T.tensor4('Data')
G_m,G_b,G_c = gpuModel.collectUpdateStatistics(H,D)
gradient = theano.function([H,D], [G_m,G_b,G_c], allow_input_downcast=True)

# gibbs sampler (up, down, sample)
D = T.tensor4('data')
[P_H,H] = gpuModel.computeHgivenV(D)
[P_V,V] = gpuModel.computeVgivenH(H)
gibbs = theano.function([D], V, allow_input_downcast=True)

print "Starting forward pass test:"
print "----------------------------"
[P_naive,S_n] = naiveModel.computeHgivenV(data)
[P_GPU,S_g] = forward(data)
print "ERROR MADE: " + str(np.mean(np.abs(P_naive-P_GPU)))
print

if np.mean(np.abs(P_naive-P_GPU)) > desired_precision:
	test_failures += 1
	print "Failed test on computing H, given V"
tests_total += 1

print "Starting backward pass test:"
print "----------------------------"
[P_V_naive, V_naive] = naiveModel.computeVgivenH(S_n)
[P_V_gpu,V_gpu] = backward(S_n)
print "ERROR MADE: " + str(np.mean(np.abs(P_V_naive-P_V_gpu)))
print

if np.mean(np.abs(P_V_naive-P_V_gpu)) > desired_precision:
	test_failures += 1
	print "Failed test on computing V, given H"
tests_total += 1

print "Starting Gradient pass test:"
print "----------------------------"
G_M_naive, G_b_naive, G_c_naive = naiveModel.collectUpdateStatistics(P_naive, data)
G_M_gpu,G_b_gpu,G_c_gpu = gradient(P_naive, data)
print "ERROR MADE (Motifs): " + str(np.mean(np.abs(G_M_naive-G_M_gpu)))
print "ERROR MADE (Bias): " + str(np.mean(np.abs(G_b_naive-G_b_gpu)))
print "ERROR MADE (c): " + str(np.mean(np.abs(G_c_naive-G_c_gpu)))

if np.mean(np.abs(G_M_naive-G_M_gpu)) > desired_precision:
	test_failures += 1
	print "Failed test on collecting update stats for kernels"
tests_total += 1

if np.mean(np.abs(G_b_naive-G_b_gpu)) > desired_precision:
	test_failures += 1
	print "Failed test on collecting update stats for bias"
tests_total += 1

if np.mean(np.abs(G_c_naive-G_c_gpu)) > desired_precision:
	test_failures += 1
	print "Failed test on collecting update stats for c"
tests_total += 1

# NOW, LOAD THE REAL DATA AND COMPARE TRAINING PROCEDURES ON THE DATA

print "Start loading real data to perform further tests..."
seqReader = dataRead.SeqReader()
allSeqs = seqReader.readSequencesFromFile('../data/wgEncodeAwgDnaseUwAg10803UniPk.fa')
realData = np.array([allSeqs[random.randrange(0, len(allSeqs))] for i in range(10)])


print "Starting Gibbs Sampling test:"
print "----------------------------"
V_naive_acc = np.zeros(realData.shape)
V_gpu_acc = np.zeros(realData.shape)
for i in range(iterations):
    V_naive = naiveModel.computeVgivenH(naiveModel.computeHgivenV(realData)[1])[1]
    V_gpu = gibbs(realData)
    V_naive_acc += V_naive
    V_gpu_acc += V_gpu
    if i % 100 == 0 and i > 0:
        print "100 iterations done"

V_naive_acc /= iterations
V_gpu_acc /= iterations
print "ERROR MADE: " + str(np.mean(np.abs(V_naive_acc-V_gpu_acc)))

if np.mean(np.abs(V_naive_acc-V_gpu_acc)) > desired_precision:
	test_failures += 1
	print "Failed test on gibbs sampling"
tests_total += 1

naiveModel = NaiveCRBM(motifLength=hyper_params['motif_length'],
                       numMotifs=hyper_params['number_of_motifs'],
                       learningRate=hyper_params['learning_rate'],
                       poolingFactor=hyper_params['pooling_factor'])

gpuModel = CRBM(hyper_params)
gpuModel.setToZero = True
# set parameters
naiveModel.setCustomKernels(kernel)
gpuModel.setCustomKernels(kernel)
gpuModel.batchSize = 1
gpuModel.debug = False
naiveModel.debug = False

naiveModel.trainModel(data, hyper_params['epochs'], hyper_params['batch_size'], hyper_params['cd_k'])
print "DONE WITH NAIVE---------"
gpuModel.trainModel(data)

new_motifs_gpu = gpuModel.motifs.get_value()
new_motifs_naive = naiveModel.kernels

print "ERROR MADE (motifs): " + str(np.mean(np.abs(new_motifs_gpu-new_motifs_naive)))

if np.mean(np.abs(new_motifs_gpu-new_motifs_naive)) > desired_precision:
	test_failures += 1
	print "Failed test on training model for several epochs"
tests_total += 1

#data = np.array([allSeqs[random.randrange(0,len(allSeqs))] for i in range(1)])
naiveModel = NaiveCRBM(motifLength=hyper_params['motif_length'],
                       numMotifs=hyper_params['number_of_motifs'],
                       learningRate=hyper_params['learning_rate'],
                       poolingFactor=hyper_params['pooling_factor'])

gpuModel = CRBM(hyper_params)
gpuModel.setToZero = True
# set parameters
naiveModel.setCustomKernels(kernel)
gpuModel.setCustomKernels(kernel)
gpuModel.batchSize = 1
gpuModel.debug = False
naiveModel.debug = False
naiveModel.updateWeights = False

# compile theano function
D = T.tensor4('data')
updates = gpuModel.updateWeightsOnMinibatch(D, 1)
der_m = updates[0][1]-updates[0][0]
der_bias = updates[1][1]-updates[1][0]
der_c = updates[2][1]-updates[2][0]
train = theano.function([D], [der_m, der_bias, der_c], allow_input_downcast=True)

precision = 10000
der_m_naive = np.zeros(kernel.shape)
der_m_gpu = np.zeros(kernel.shape)
der_bias_naive = np.zeros(naiveModel.bias.shape)
der_bias_gpu = np.zeros(gpuModel.bias.get_value().shape)
der_c_naive = np.zeros(naiveModel.c.shape)
der_c_gpu = np.zeros(gpuModel.c.get_value().shape)

for i in range(precision):
    # naive
    [der_m_naive_l, der_bias_naive_l, der_c_naive_l] = naiveModel.updateWeightsOnMinibatch(data, 1)
    der_m_naive += der_m_naive_l
    der_bias_naive += der_bias_naive_l
    der_c_naive += der_c_naive_l
    # gpu
    [der_m_gpu_l, der_bias_gpu_l, der_c_gpu_l] = train(data)
    der_m_gpu += der_m_gpu_l
    der_bias_gpu += der_bias_gpu_l
    der_c_gpu += der_c_gpu_l
    

der_m_naive /= precision
der_bias_naive /= precision
der_c_naive /= precision

der_m_gpu /= precision
der_bias_gpu /= precision
der_c_gpu /= precision

print
print "ERROR MADE (motifs): " + str(np.mean(np.abs(der_m_naive - der_m_gpu)))
print "ERROR MADE (bias): " + str(np.mean(np.abs(der_bias_naive - der_bias_gpu)))
print "ERROR MADE (c): " + str(np.mean(np.abs(der_c_naive - der_c_gpu)))
print

if np.mean(np.abs(der_m_naive - der_m_gpu)) > desired_precision:
	test_failures += 1
	print "Failed test on calculating the right derivatives (motifs)"
tests_total += 1

if np.mean(np.abs(der_bias_naive - der_bias_gpu)) > desired_precision:
	test_failures += 1
	print "Failed test on calculating the right derivatives (bias)"
tests_total += 1

if np.mean(np.abs(der_c_naive - der_c_gpu)) > desired_precision:
	test_failures += 1
	print "Failed test on calculating the right derivatives (c)"
tests_total += 1


print "-----------------------"
print "STATISTICS FROM TESTING"
print
print "Tests performed: " + str(tests_total)
print "Successfull tests: " + str(tests_total-test_failures)
print "Failures: " + str(test_failures)
print "Percentage of success: " + str((tests_total-test_failures) / float(tests_total))

