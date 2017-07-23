import numpy as np
from Bio import SeqIO
from Bio import motifs
from Bio.Alphabet import IUPAC
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append("../code")
from getData import seqToOneHot, readSeqsFromFasta
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
import numpy as np
from convRBM import CRBM


outputdir = os.environ["CRBM_OUTPUT_DIR"]
def getFeatures(record, seqs):
    flen = len(seqs[0]) - record[0].length + 1
    features = np.zeros((len(seqs), flen, len(record)))
    # load motifs as pwms
    for m in range(len(record)):
        r = record[m]
        r.pseudocounts =1
        pssm = r.pssm
        for s in range(len(seqs)):
            features[s,:,m] += pssm.calculate(seqs[s].seq)
            features[s,:,m] += pssm.reverse_complement().calculate(seqs[s].seq)
    return features

def appendPlt(labels, score, name):
    #print("auc: "+str(auc))
    auc=roc_auc_score(labels,score)
    print(auc)
    fpr, tpr, _ = roc_curve(labels,score)
    plt.plot(fpr, tpr, label = "{}: {:1.3f}".format(name, auc))

def evalObjective(bias, score, labels):
    trf = (1/(1+np.exp(-score - bias))).sum(axis=(1))
    lr = LogisticRegression()
    lr.fit(trf, labels)
    pred = lr.decision_function(trf)
    return 1-roc_auc_score(labels, pred)

#from scipy.optimize import  minimize
def getBestBias(features, labels):
    biases = np.zeros(tr_features.shape[2])
    brange = np.linspace(0, 60, num=50*5 + 1)
    for i in range(tr_features.shape[2]):
        minres = np.inf
        for b in brange:
            res = evalObjective(b, tr_features[:,:,[i]],labels)
            #print(res)
            if res < minres:
                biases[i] = b
                minres = res
        print(biases[i])
    return biases

# first analyse how well sequences can be discriminated
# based on MEME derived motifs

tr_o = readSeqsFromFasta('../data/stemcells_train.fa')
te_o = readSeqsFromFasta('../data/stemcells_test.fa')
tr_m = readSeqsFromFasta('../data/fibroblast_train.fa')
te_m = readSeqsFromFasta('../data/fibroblast_test.fa')

trset = tr_o + tr_m
teset = te_o + te_m
trl = np.concatenate((np.ones(len(tr_o)), np.zeros(len(tr_m))))
tel = np.concatenate((np.ones(len(te_o)), np.zeros(len(te_m))))

# obtain motifs from meme run
records = []
mfiles = [outputdir + '/meme_oct4/meme_out/meme.txt',
        outputdir + '/meme_mafk/meme_out/meme.txt']
for mfile in mfiles:
    f = open(mfile, "r")
    record = motifs.parse(f, "meme")
    records += record

tr_features = getFeatures(records, trset)
te_features = getFeatures(records, teset)

fig = plt.figure(figsize = (5,4))
alphas = range(1,9)

#beta optimization search for the best beta

biases = getBestBias(tr_features, trl)
trf = (1/(1+np.exp(-tr_features - biases))).sum(axis=1)
tef = (1/(1+np.exp(-te_features - biases))).sum(axis=1)
lr = LogisticRegression()
lr.fit(trf, trl)
pred = lr.decision_function(trf)
print("auc: {:1.3f}".format(roc_auc_score(trl, pred)))
print("auPRC: {:1.3f}".format(average_precision_score(trl, pred)))
pred = lr.decision_function(tef)
print("auc: {:1.3f}".format(roc_auc_score(tel, pred)))
print("auPRC: {:1.3f}".format(average_precision_score(tel, pred)))
appendPlt(tel, pred, "MEME")

# generate cRBM models
crbm_oct4 = CRBM.loadModel(outputdir + "/oct4_model.pkl")
crbm_mafk = CRBM.loadModel(outputdir + "/mafk_model.pkl")

tr_o = seqToOneHot(readSeqsFromFasta('../data/stemcells_train.fa'))
te_o = seqToOneHot(readSeqsFromFasta('../data/stemcells_test.fa'))
tr_m = seqToOneHot(readSeqsFromFasta('../data/fibroblast_train.fa'))
te_m = seqToOneHot(readSeqsFromFasta('../data/fibroblast_test.fa'))

te_merged = np.concatenate( (te_o, te_m), axis=0 )
tr_merged = np.concatenate( (tr_o, tr_m), axis=0 )


trf = np.concatenate((crbm_oct4.getHitProbs(tr_merged).sum(axis=(2,3)),
        crbm_mafk.getHitProbs(tr_merged).sum(axis=(2,3))), axis =1)
tef = np.concatenate((crbm_oct4.getHitProbs(te_merged).sum(axis=(2,3)),
        crbm_mafk.getHitProbs(te_merged).sum(axis=(2,3))), axis =1)

trl = np.concatenate( (np.ones(tr_o.shape[0]), \
        np.zeros(tr_m.shape[0])), axis=0 )
tel = np.concatenate( (np.ones(te_o.shape[0]), \
        np.zeros(te_m.shape[0])), axis=0 )

lr = LogisticRegression()
lr.fit(trf, trl)
pred = lr.decision_function(tef)
print("")
print("auc: {:1.3f}".format(roc_auc_score(tel, pred)))
print("auPRC: {:1.3f}".format(average_precision_score(tel, pred)))
appendPlt(tel, pred, "cRBM-all")


#createLogo(crbm_oct4.getPFMs()[2], "Oct4-like1")
#createLogo(crbm_oct4.getPFMs()[5], "Oct4-like2")
#createLogo(crbm_mafk.getPFMs()[9], "Mafk-like1")
#createLogo(crbm_mafk.getPFMs()[3], "Mafk-like2")
# evaluate free energy for testing data
#print "Get free energy for both models..."
#score_oct4 = crbm_oct4.freeEnergy(te_merged)
#score_mafk = crbm_mafk.freeEnergy(te_merged)
#score = score_mafk - score_oct4
trf = np.concatenate((crbm_oct4.getHitProbs(tr_merged).sum(axis=(2,3))[:,[2,5]],
    crbm_mafk.getHitProbs(tr_merged).sum(axis=(2,3))[:,[3,9]]), axis =1)
tef = np.concatenate((crbm_oct4.getHitProbs(te_merged).sum(axis=(2,3))[:,[2,5]],
    crbm_mafk.getHitProbs(te_merged).sum(axis=(2,3))[:,[3,9]]), axis =1)

lr = LogisticRegression()
lr.fit(trf, trl)
pred = lr.decision_function(tef)
print("")
print("auc: {:1.3f}".format(roc_auc_score(tel, pred)))
print("auPRC: {:1.3f}".format(average_precision_score(tel, pred)))

appendPlt(tel, pred, "cRBM-reduced")

plt.legend(loc="lower right")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
fig.savefig(outputdir + "auroc_crbm_meme.eps", dpi=700)

#finishPlt(outputdir + "auroc_crbm_meme.eps")







#analyse the mixed sequence experiments


trset = readSeqsFromFasta('../data/joint_train.fa')
teset = readSeqsFromFasta('../data/joint_test.fa')

trl = np.concatenate((np.ones(len(tr_o)), np.zeros(len(tr_m))))
tel = np.concatenate((np.ones(len(te_o)), np.zeros(len(te_m))))

# obtain motifs from meme run
records = []
mfiles = [outputdir + '/meme_joint_om/meme_out/meme.txt']
for mfile in mfiles:
    f = open(mfile, "r")
    record = motifs.parse(f, "meme")
    records += record

tr_features = getFeatures(records, trset)
te_features = getFeatures(records, teset)

fig = plt.figure(figsize = (5,4))
alphas = range(1,9)

biases = getBestBias(tr_features, trl)
trf = (1/(1+np.exp(-tr_features - biases))).sum(axis=1)
tef = (1/(1+np.exp(-te_features - biases))).sum(axis=1)
lr = LogisticRegression()
lr.fit(trf, trl)
pred = lr.decision_function(trf)
print("Percentile {:d}".format(alpha))
print("auc: {:1.3f}".format(roc_auc_score(trl, pred)))
print("auPRC: {:1.3f}".format(average_precision_score(trl, pred)))
pred = lr.decision_function(tef)
print("auc: {:1.3f}".format(roc_auc_score(tel, pred)))
print("auPRC: {:1.3f}".format(average_precision_score(tel, pred)))
appendPlt(tel, pred, "MEME")

# generate cRBM models
crbm = CRBM.loadModel(outputdir + "/oct4_mafk_joint_model.pkl")

tr_o = seqToOneHot(readSeqsFromFasta('../data/stemcells_train.fa'))
te_o = seqToOneHot(readSeqsFromFasta('../data/stemcells_test.fa'))
tr_m = seqToOneHot(readSeqsFromFasta('../data/fibroblast_train.fa'))
te_m = seqToOneHot(readSeqsFromFasta('../data/fibroblast_test.fa'))

te_merged = np.concatenate( (te_o, te_m), axis=0 )
tr_merged = np.concatenate( (tr_o, tr_m), axis=0 )

trf = crbm.getHitProbs(tr_merged).sum(axis=(2,3))
tef = crbm.getHitProbs(te_merged).sum(axis=(2,3))

trl = np.concatenate( (np.ones(tr_o.shape[0]), \
        np.zeros(tr_m.shape[0])), axis=0 )
tel = np.concatenate( (np.ones(te_o.shape[0]), \
        np.zeros(te_m.shape[0])), axis=0 )

lr = LogisticRegression()
lr.fit(trf, trl)
pred = lr.decision_function(tef)
print("")
print("auc: {:1.3f}".format(roc_auc_score(tel, pred)))
print("auPRC: {:1.3f}".format(average_precision_score(tel, pred)))
appendPlt(tel, pred, "cRBM-all")


trf = crbm.getHitProbs(tr_merged).sum(axis=(2,3))[:,[1,8]]
tef = crbm.getHitProbs(te_merged).sum(axis=(2,3))[:,[1,8]]

lr = LogisticRegression()
lr.fit(trf, trl)
pred = lr.decision_function(tef)
print("")
print("auc: {:1.3f}".format(roc_auc_score(tel, pred)))
print("auPRC: {:1.3f}".format(average_precision_score(tel, pred)))

appendPlt(tel, pred, "cRBM-reduced")


plt.legend(loc="lower right")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
fig.savefig(outputdir + "auroc_crbm_meme_joint.eps", dpi=700)


