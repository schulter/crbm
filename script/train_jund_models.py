import matplotlib.pyplot as plt
import numpy as np
import random
import time
from datetime import datetime
import joblib
import os
import sys
sys.path.append("../code")

from convRBM import CRBM
import plotting
from getData import loadSequences

outputdir = os.environ["CRBM_OUTPUT_DIR"]

########################################################
# SET THE HYPER PARAMETERS
hyperParams={
        'number_of_motifs': 10,
        'motif_length': 15,
        'input_dims':4,
        'momentum':0.95,
        'learning_rate': .5,
        'doublestranded': True,
        'pooling_factor': 1,
        'epochs': 300,
        'cd_k': 5,
        'sparsity': 0.5,
        'batch_size': 100,
    }

np.set_printoptions(precision=4)
train_test_ratio = 0.1
batchSize = hyperParams['batch_size']
np.set_printoptions(precision=4)
########################################################

# get the data

#cells = ["HepG2", "HCT116", "MCF-7", "GM12878", "A549"]
cells = ["HepG2_ENCFF002CUF", "K562_ENCFF001VRL", \
    "HeLa-S3_ENCFF001VIQ", "GM12878_ENCFF002COV", \
            "H1hesc_ENCFF002CIZ"]
training = []
test = []

for cell in cells:
    tr, te = loadSequences(
        '../data/jund_data/{}.fasta'.format(cell), \
            train_test_ratio, 2000)
    training.append(tr)
    test.append(te)

crbm = CRBM(hyperParams=hyperParams)

test_merged = np.concatenate( test, axis=0 )

training_merged = np.concatenate( training, axis=0 )

# clip the sequences if necessary
nseq=int((test_merged.shape[3]-hyperParams['motif_length'] + \
        1)/hyperParams['pooling_factor'])*\
                hyperParams['pooling_factor']+ hyperParams['motif_length'] -1

test_merged = test_merged[:,:,:,:nseq]
training_merged =training_merged[:,:,:,:nseq]

print "Train cRBM ..."
start = time.time()
crbm.trainModel(training_merged, test_merged)

crbm.saveModel(outputdir + "/joint_jund_model.pkl")

joblib.dump((training, test), outputdir + "jund_dataset.pkl")

