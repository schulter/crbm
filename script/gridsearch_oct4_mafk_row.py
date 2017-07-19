import numpy as np
import pandas as pd
import os
import sys
from sklearn import metrics
sys.path.append("../code")

from convRBM import CRBM
from getData import seqToOneHot, readSeqsFromFasta

outputdir = os.environ["CRBM_OUTPUT_DIR"] + "/grom/"
if not os.path.exists(outputdir):
    os.mkdir(outputdir)

print(sys.argv)
lr = float(sys.argv[1])
lam = float(sys.argv[2])
bs = int(sys.argv[3])
rho = float(sys.argv[4])
spm = sys.argv[5]
ep = int(sys.argv[6])

filename = "_".join(sys.argv) + ".csv"
print(outputdir + filename)

# get the data

tr_o = seqToOneHot(readSeqsFromFasta('../data/stemcells_train.fa'))
te_o = seqToOneHot(readSeqsFromFasta('../data/stemcells_test.fa'))
tr_m = seqToOneHot(readSeqsFromFasta('../data/fibroblast_train.fa'))
te_m = seqToOneHot(readSeqsFromFasta('../data/fibroblast_test.fa'))

te_merged = np.concatenate( (te_o, te_m), axis=0 )
tr_merged = np.concatenate( (tr_o, tr_m), axis=0 )
labels = np.concatenate( (np.ones(te_o.shape[0]), \
        np.zeros(te_m.shape[0])), axis=0 )

results = pd.DataFrame(np.zeros((1, 7)),\
        columns = ["LearningRate", "Sparsity", "Batchsize", "rho", "spmethod", \
        "epochs", "auROC"])

# generate cRBM models
# train model
print "Train cRBM with " + filename
crbm_stem = CRBM(num_motifs=10, motif_length=15, \
        learning_rate = lr, lambda_rate = lam,
        batchsize = bs, rho = rho, epochs = ep, spmethod = spm)
crbm_fibro = CRBM(num_motifs=10, motif_length=15, \
        learning_rate = lr, lambda_rate = lam,
        batchsize = bs, rho = rho, epochs = ep, spmethod = spm)

idx=0
crbm_stem.fit(tr_o,te_o)
crbm_fibro.fit(tr_m,te_o)

score_sc = crbm_stem.freeEnergy(te_merged)
score_fibro = crbm_fibro.freeEnergy(te_merged)
score = score_fibro - score_sc
auc=metrics.roc_auc_score(labels,score)

print("auc: "+str(auc))
results["LearningRate"].iloc[idx]=lr
results["Sparsity"].iloc[idx]=lam
results["Batchsize"].iloc[idx]=bs
results["rho"].iloc[idx]=rho
results["spmethod"].iloc[idx]=spm
results["epochs"].iloc[idx]=ep
results["auPRC"].iloc[idx]=prc
results["auROC"].iloc[idx]=auc

print("Finished " + outputdir+filename)
results.to_csv(outputdir + filename, sep="\t", index=False, header = False)

