import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import random
import time
from datetime import datetime
import joblib
import os
import sys
from sklearn import metrics
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
        'learning_rate': .1,
        'doublestranded': True,
        'pooling_factor': 1,
        'rho': .01,
        'epochs': 100,
        'cd_k': 5,
        'sparsity': 0.5,
        'batch_size': 20,
    }

lrates = [0.01, 0.05, 0.1, 0.5, 1.]
sparsities = [0.0, 0.01, 0.1, 0.5, 1., 10]
batchsizes = [10, 50, 100, 500]
rhos = [0.0001, 0.001, 0.01, 0.1]

np.set_printoptions(precision=4)
train_test_ratio = 0.1
np.set_printoptions(precision=4)
########################################################

# get the data
cells = ["GM12878_ENCFF002COV", "H1hesc_ENCFF001UBM"]
tr_gm, te_gm = loadSequences(
    '../data/jund_data/{}.fasta'.format(cells[0]), \
        train_test_ratio)
tr_h1, te_h1 = loadSequences(
    '../data/jund_data/{}.fasta'.format(cells[1]), \
        train_test_ratio)

test_merged = np.concatenate( (te_gm, te_h1), axis=0 )

labels = np.concatenate( (np.ones(te_gm.shape[0]), \
        np.zeros(te_h1.shape[0])), axis=0 )

results = pd.DataFrame(np.zeros((len(lrates)*len(sparsities)*\
        len(batchsizes)*len(rhos), 6)),\
        columns = ["LearningRate", "Sparsity", "Batchsize", "rho", "auPRC", "auROC"])

idx=0
for par in itertools.product(lrates, sparsities, batchsizes, rhos):
    # generate cRBM models
    # train model
    print "Train cRBM with lr={:1.2f}, sp={:1.3f}, bs={:3.0f}, rho={:1.4f}".format( \
        par[0], par[1], par[2], par[3])
    hyperParams['learning_rate'] =par[0]
    hyperParams['sparsity'] =par[1]
    hyperParams['batch_size'] =par[2]
    hyperParams['rho'] =par[3]
    crbm_gm = CRBM(hyperParams=hyperParams)
    crbm_h1 = CRBM(hyperParams=hyperParams)
    crbm_gm.trainModel(tr_gm,te_gm)
    crbm_h1.trainModel(tr_h1,te_gm)
    score_gm = crbm_gm.freeEnergy(test_merged)
    score_h1 = crbm_h1.freeEnergy(test_merged)
    score = score_h1 - score_gm
    auc=metrics.roc_auc_score(labels,score)
    print("auc: "+str(auc))
    prc=metrics.average_precision_score(labels,score)
    print("prc: "+str(prc))
    results["LearningRate"].iloc[idx]=par[0]
    results["Sparsity"].iloc[idx]=par[1]
    results["Batchsize"].iloc[idx]=par[2]
    results["rho"].iloc[idx]=par[3]
    results["auPRC"].iloc[idx]=prc
    results["auROC"].iloc[idx]=auc
    idx += 1

results.to_csv(outputdir + "jund_gridsearch.csv", sep="\t", index=None)

