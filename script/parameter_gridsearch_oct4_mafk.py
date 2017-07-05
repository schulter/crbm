import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        'epochs': 50,
        'cd_k': 5,
        'sparsity': 0.5,
        'batch_size': 20,
    }

lrates = [0.01, 0.05, 0.1, 0.5, 1.]
sparsities = [0.0, 0.01, 0.1, 0.5, 1., 10]
batchsizes = [10, 50, 100, 500]

np.set_printoptions(precision=4)
train_test_ratio = 0.1
np.set_printoptions(precision=4)
########################################################

# get the data

training_stem, test_stem = loadSequences('../data/stemcells.fa', \
        train_test_ratio)
training_fibro, test_fibro = loadSequences('../data/fibroblast.fa', \
        train_test_ratio, 4000)

test_merged = np.concatenate( (test_stem, test_fibro), axis=0 )
training_merged = np.concatenate( (training_stem, training_fibro), axis=0 )
nseq=int((test_merged.shape[3]-hyperParams['motif_length'] + \
        1)/hyperParams['pooling_factor'])*\
                hyperParams['pooling_factor']+ hyperParams['motif_length'] -1
test_merged = test_merged[:,:,:,:nseq]
training_merged =training_merged[:,:,:,:nseq]

labels = np.concatenate( (np.ones(test_stem.shape[0]), \
        np.zeros(test_fibro.shape[0])), axis=0 )

results = pd.DataFrame(np.zeros((len(lrates)*len(sparsities)*len(batchsizes), 5)),\
        columns = ["LearningRate", "Sparsity", "Batchsize", "auPRC", "auROC"])

idx=0
for par in itertools.product(lrates, sparsities, batchsizes):
    # generate cRBM models
    # train model
    print "Train cRBM with lr={:1.2f}, sp={:1.3f}, bs={:3.0f}".format( \
        par[0], par[1], par[2])
    hyperParams['learning_rate'] =par[0]
    hyperParams['sparsity'] =par[1]
    hyperParams['batch_size'] =par[2]
    crbm_stem = CRBM(hyperParams=hyperParams)
    crbm_fibro = CRBM(hyperParams=hyperParams)
    crbm_stem.trainModel(training_stem,test_stem)
    crbm_fibro.trainModel(training_fibro,test_fibro)
    score_sc = crbm_stem.freeEnergy(test_merged)
    score_fibro = crbm_fibro.freeEnergy(test_merged)
    score = score_fibro - score_sc
    auc=metrics.roc_auc_score(labels,score)
    print("auc: "+str(auc))
    prc=metrics.average_precision_score(labels,score)
    print("prc: "+str(prc))
    results["LearningRate"].iloc[idx]=par[0]
    results["Sparsity"].iloc[idx]=par[1]
    results["Batchsize"].iloc[idx]=par[2]
    results["auPRC"].iloc[idx]=prc
    results["auROC"].iloc[idx]=auc
    idx += 1

results.to_csv(outputdir + "om_gridsearch.csv", sep="\t", index=None)

