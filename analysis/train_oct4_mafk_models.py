import matplotlib.pyplot as plt
import numpy as np
import random
import time
from datetime import datetime
import joblib
import os
import sys
from sklearn import metrics

from crbm import CRBM, seqToOneHot, readSeqsFromFasta

outputdir = os.environ["CRBM_OUTPUT_DIR"]

# get the data

tr_o = seqToOneHot(readSeqsFromFasta('data/oct4_mafk_data/stemcells_train.fa'))
te_o = seqToOneHot(readSeqsFromFasta('data/oct4_mafk_data/stemcells_test.fa'))
tr_m = seqToOneHot(readSeqsFromFasta('data/oct4_mafk_data/fibroblast_train.fa'))
te_m = seqToOneHot(readSeqsFromFasta('data/oct4_mafk_data/fibroblast_test.fa'))

te_merged = np.concatenate( (te_o, te_m), axis=0 )
tr_merged = np.concatenate( (tr_o, tr_m), axis=0 )
labels = np.concatenate( (np.ones(te_o.shape[0]), \
        np.zeros(te_m.shape[0])), axis=0 )


# generate cRBM models
crbm_oct4 = CRBM(num_motifs = 10, motif_length = 15, epochs = 300)
crbm_mafk = CRBM(num_motifs = 10, motif_length = 15, epochs = 300)

# train model
print "Train cRBM ..."
crbm_oct4.fit(tr_o,te_o)
crbm_mafk.fit(tr_m,te_m)


# evaluate free energy for testing data
print "Get free energy for both models..."
score_oct4 = crbm_oct4.freeEnergy(te_merged)
score_mafk = crbm_mafk.freeEnergy(te_merged)
score = score_mafk - score_oct4

auc=metrics.roc_auc_score(labels,score)
print("auc: "+str(auc))
prc=metrics.average_precision_score(labels,score)
print("prc: "+str(prc))
crbm_oct4.printHyperParams()

crbm_merged = CRBM(num_motifs = 10, motif_length = 15, epochs = 300)
crbm_merged.fit(tr_merged, te_merged)

crbm_oct4.saveModel(outputdir + "/oct4_model.pkl")
crbm_mafk.saveModel(outputdir + "/mafk_model.pkl")
crbm_merged.saveModel(outputdir + "/oct4_mafk_joint_model.pkl")

