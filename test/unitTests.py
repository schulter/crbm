# Imports
import sys
import os
import random
import time
import numpy as np
import theano
import theano.tensor as T
#np.set_printoptions(precision=2, suppress=True)

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

test_failures = 0
tests_total = 0
desired_precision = 0.0001
iterations = 1000

seqReader = dataRead.SeqReader()
print "Loading dataset ..."
data = seqReader.readOneHotFromFile('../data/seq.onehot.gz')

per=np.random.permutation(len(data))
isubset=per[:3]

data=np.array([data[i] for i in isubset])

test_names=["Single stranded tests without pooling",
		"Single stranded tests pooling 5 nucleotides",
		"Double stranded tests without pooling",
		"Double stranded tests pooling 5 nucleotides"]
test_params=[
{'number_of_motifs':2,
                'motif_length':11,
                'learning_rate':0.1,
                'pooling_factor':1,
                'epochs':100,
                'cd_k':1,
                'batch_size':100,
								'sparsity':0.1,
								'rho':0.001,
								'verbose':False,
								'momentum':0.9,
                'doublestranded':False
},
{'number_of_motifs':2,
                'motif_length':11,
                'learning_rate':0.1,
                'pooling_factor':5,
                'epochs':100,
                'cd_k':1,
                'batch_size':100,
								'sparsity':0.1,
								'rho':0.001,
								'verbose':False,
								'momentum':0.9,
                'doublestranded':False
},
{'number_of_motifs':2,
                'motif_length':11,
                'learning_rate':0.1,
                'pooling_factor':1,
                'epochs':100,
                'cd_k':1,
                'batch_size':100,
								'sparsity':0.1,
								'rho':0.001,
								'verbose':False,
								'momentum':0.9,
                'doublestranded':True
},
{'number_of_motifs':2,
                'motif_length':11,
                'learning_rate':0.1,
                'pooling_factor':5,
                'epochs':100,
                'cd_k':1,
                'batch_size':100,
								'sparsity':0.1,
								'rho':0.001,
								'verbose':False,
								'momentum':0.9,
                'doublestranded':True
}]

for i in range(len(test_names)):
	print("-----------")
	print(test_names[i])
	print("-----------")
	print
	#initialize the learner and set custom kernels
	hyper_params = test_params[i]

	gpuModel = CRBM(hyper_params)
	naiveModel = NaiveCRBM(hyper_params)

	naiveModel.setMotifs(gpuModel.motifs.get_value())
	naiveModel.setBiases(gpuModel.bias.get_value())
	naiveModel.setC(gpuModel.c.get_value())

	# create theano functions
	# forward
	print "Compiling theano functions..."
	D = T.tensor4("data")
	H = T.tensor4()

	out = gpuModel.computeHgivenV(D)
	computeHgivenVFast = theano.function([D], out, allow_input_downcast=True)

	# backward
	out = gpuModel.computeVgivenH(H)
	computeVgivenHFast = theano.function([H], out, allow_input_downcast=True)

	#collect VH statistics
	vhstat = gpuModel.collectVHStatistics(H,D)
	vhStatFast = theano.function([H,D], vhstat, allow_input_downcast=True)

	#collect H statistics
	hstat = gpuModel.collectHStatistics(D)
	hStatFast = theano.function([D], hstat, allow_input_downcast=True)

	#collect H statistics
	vstat = gpuModel.collectVStatistics(D)
	vStatFast = theano.function([D], vstat, allow_input_downcast=True)

	#update function
	updates = gpuModel.updateWeightsOnMinibatch(D,hyper_params['cd_k'])
	
	trainingFun = theano.function(
              [D], None, updates = updates,
              allow_input_downcast=True,
              name='train_CRBM')
	print "Done!"
	[ph_slow,h] = naiveModel.computeHgivenV(data)
	[ph_fast,h] = computeHgivenVFast(data)
	print


	if np.mean(np.abs(ph_slow-ph_fast)) >= desired_precision:
			test_failures += 1
			print("Test computeHgivenV: FAILED")
			print ph_slow
			print ph_fast
	else:
			print("Test computeHgivenV: PASSED")
			tests_total += 1

	print("----------------------------")

	[pv_slow, v] = naiveModel.computeVgivenH(h)
	[pv_fast,v] = computeVgivenHFast(h)

	if np.mean(np.abs(pv_slow-pv_fast)) >= desired_precision:
		test_failures += 1
		print "Test computeVgivenH: FAILED"
		print pv_slow
		print pv_fast
	else:
		print "Test computeVgivenH: PASSED"
		tests_total += 1
	print "----------------------------"

	vh_slow = naiveModel.collectVHStatistics(ph_fast, data)
	vh_fast = vhStatFast(ph_fast, data)

	if np.mean(np.abs(vh_slow-vh_fast)) >= desired_precision:
		test_failures += 1
		print "Test VH Statistics: FAILED"
		print vh_slow
		print vh_fast
	else:
		print "Test VH Statistics: PASSED"
		tests_total += 1
	print "----------------------------"

	h_slow = naiveModel.collectHStatistics(ph_fast)
	h_fast = hStatFast(ph_fast)

	if np.mean(np.abs(h_slow-h_fast)) >= desired_precision:
		test_failures += 1
		print "Test H Statistics: FAILED"
		print h_slow
		print h_fast
	else:
		print "Test H Statistics: PASSED"
		tests_total += 1
	print "----------------------------"

	v_slow = naiveModel.collectVStatistics(v)
	v_fast = vStatFast(v)

	if np.mean(np.abs(v_slow-v_fast)) >= desired_precision:
		test_failures += 1
		print "Test V Statistics: FAILED"
		print v_slow
		print v_fast
	else:
		print "Test V Statistics: PASSED"
		tests_total += 1
	print "----------------------------"
	if hyper_params['doublestranded']:
		i=range(0,2*hyper_params['number_of_motifs'],2)
		j=range(1,2*hyper_params['number_of_motifs'],2)
		if np.mean(np.abs(naiveModel.bias[0,i]-naiveModel.bias[0,j])) >=desired_precision:
			test_failures += 1
			print "Test matching of initial biases: FAILED"
			print h_fast[0,i]
			print h_fast[0,j]
		else:
			print "Test matching of initial biases: PASSED"
			tests_total += 1
		print "----------------------------"
		if np.mean(np.abs(naiveModel.motifs[i,0,:,:]-naiveModel.motifs[j,0,::-1,::-1])) >= desired_precision:
			test_failures += 1
			print "Test matching of initial motifs: FAILED"
			print vh_fast[i,0,:,:]
			print vh_fast[j,0,::-1,::-1]
		else:
			print "Test matching of initial motifs: PASSED"
			tests_total += 1
		print "----------------------------"
		trainingFun(data)
		if np.mean(np.abs(gpuModel.bias.get_value()[0,i]-gpuModel.bias.get_value()[0,j])) >=desired_precision:
			test_failures += 1
			print "Test matching of updated biases: FAILED"
			print h_fast[0,i]
			print h_fast[0,j]
		else:
			print "Test matching of updated biases: PASSED"
			tests_total += 1
		print "----------------------------"
		if np.mean(np.abs(gpuModel.motifs.get_value()[i,0,:,:]-gpuModel.motifs.get_value()[j,0,::-1,::-1])) >= desired_precision:
			test_failures += 1
			print "Test matching of updated motifs: FAILED"
			print vh_fast[i,0,:,:]
			print vh_fast[j,0,::-1,::-1]
		else:
			print "Test matching of updated motifs: PASSED"
			tests_total += 1
		print "----------------------------"


