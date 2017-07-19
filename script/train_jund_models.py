import matplotlib.pyplot as plt
import numpy as np
import random
import time
from datetime import datetime
from sklearn import metrics
import joblib
import os
import sys
sys.path.append("../code")

from convRBM import CRBM
from getData import seqToOneHot, readSeqsFromFasta

outputdir = os.environ["CRBM_OUTPUT_DIR"]

# get the data
cells = ["GM12878", "H1hesc", "Hela", "HepG2", "K562"]
trsets = []
tesets = []

for cell in cells:
    trsets.append(seqToOneHot(readSeqsFromFasta(
    '../data/jund_data/' + cell + '_only_train.fa')))
    tesets.append(seqToOneHot(readSeqsFromFasta(
    '../data/jund_data/' + cell + '_only_test.fa')))

    crbm = CRBM(num_motifs = 10, motif_length = 15, epochs = 300)
    crbm.fit(trsets[-1], trsets[-1])
    crbm.saveModel(outputdir + "/jund_{}.pkl".format(cell))


te_merged = np.concatenate( tesets, axis=0 )
tr_merged = np.concatenate( trsets, axis=0 )


crbm = CRBM(num_motifs = 10, motif_length = 15, epochs = 300)
crbm.fit(tr_merged, te_merged)
crbm.saveModel(outputdir + "/jund_joint.pkl")

