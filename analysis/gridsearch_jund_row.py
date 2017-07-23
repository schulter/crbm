import numpy as np
import pandas as pd
import os
import sys
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
sys.path.append("../code")

from convRBM import CRBM
from getData import seqToOneHot, readSeqsFromFasta

outputdir = os.environ["CRBM_OUTPUT_DIR"] + "/grjund/"
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
# get the data
cells = ["GM12878", "H1hesc", "HeLa-S3", "HepG2", "K562"]
trsets = []
tesets = []

def getOneHot(dataset):
    l = np.concatenate([ np.repeat(i, dataset[i].shape[0]) for i in range(len(dataset)) ])
    eye = np.eye(len(dataset))
    oh = np.asarray([ eye[:,i] for i in l])
    return oh

def getLabels(dataset):
    l = np.concatenate([ np.repeat(i, dataset[i].shape[0]) for i in range(len(dataset)) ])
    return l


for cell in cells:
    trsets.append(seqToOneHot(readSeqsFromFasta(
    '../data/jund_data/' + cell + '_only_train.fa')))
    tesets.append(seqToOneHot(readSeqsFromFasta(
    '../data/jund_data/' + cell + '_only_test.fa')))


te_merged = np.concatenate( tesets, axis=0 )
tr_merged = np.concatenate( trsets, axis=0 )


results = pd.DataFrame(np.zeros((1, 7)),\
        columns = ["LearningRate", "Sparsity", "Batchsize", "rho", "spmethod", \
        "epochs", "Loss"])

# generate cRBM models
# train model
print "Train cRBM with " + filename
crbm = CRBM(num_motifs=10, motif_length=15, \
        learning_rate = lr, lambda_rate = lam,
        batchsize = bs, rho = rho, epochs = ep, spmethod = spm)

idx=0
crbm.fit(tr_merged,te_merged)

trf = crbm.getHitProbs(tr_merged).sum(axis=(2,3))
tef = crbm.getHitProbs(te_merged).sum(axis=(2,3))


trl = getLabels(trsets)
tel = getLabels(tesets)

lrmodel = LogisticRegression(multi_class='multinomial', solver='newton-cg')

lrmodel.fit(trf, trl)

pred = lrmodel.predict_log_proba(tef)

# compute the loss after training
loss = -np.mean(getOneHot(tesets) * pred)
#auc=metrics.roc_auc_score(tel,pred)
print("loss: {:1.3f}".format(loss))

results["LearningRate"].iloc[idx]=lr
results["Sparsity"].iloc[idx]=lam
results["Batchsize"].iloc[idx]=bs
results["rho"].iloc[idx]=rho
results["spmethod"].iloc[idx]=spm
results["epochs"].iloc[idx]=ep
results["Loss"].iloc[idx]=loss

print("Finished " + outputdir+filename)
results.to_csv(outputdir + filename, sep="\t", index=False, header = False)

