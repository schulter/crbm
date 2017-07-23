import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import random
import time
from datetime import datetime
from sklearn import metrics
import joblib
import os
import sys
from sklearn.linear_model import LogisticRegression
sys.path.append("../code")

from convRBM import CRBM
from getData import seqToOneHot, readSeqsFromFasta
import plotting

outputdir = os.environ["CRBM_OUTPUT_DIR"]

def getOneHot(dataset):
    l = np.concatenate([ np.repeat(i, dataset[i].shape[0]) for i in range(len(dataset)) ])
    eye = np.eye(len(dataset))
    oh = np.asarray([ eye[:,i] for i in l])
    return oh

def getLabels(dataset):
    l = np.concatenate([ np.repeat(i, dataset[i].shape[0]) for i in range(len(dataset)) ])
    return l

def plotConfusionMatrix(cells, labels, predictions, filename = None):
    cm = metrics.confusion_matrix(labels, predictions)
    cm = cm.astype("float32") / cm.sum(axis=1)[:, np.newaxis]
    df = pd.DataFrame(cm, columns = cells, index = cells)
    import seaborn as sns
    sns.set(style="white")
    # Generate a mask for the upper triangle
    f, ax = plt.subplots(figsize=(11, 11))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(df, cmap=cmap,  
            square=True, annot= True, linewidths=.5, cbar_kws={"shrink": .5}, 
            ax=ax)
    ax.set_ylabel("True cell type")
    ax.set_xlabel("Predicted cell type")
    #plt.set_title(name)
    if filename:
        f.savefig(filename, dpi = 1000)
    else:
        plt.show()

# get the data
cells = ["GM12878", "H1hesc", "HeLa-S3", "HepG2", "K562"]
trsets = []
tesets = []

for cell in cells:
    trsets.append(seqToOneHot(readSeqsFromFasta(
    '../data/jund_data/' + cell + '_only_train.fa')))
    tesets.append(seqToOneHot(readSeqsFromFasta(
    '../data/jund_data/' + cell + '_only_test.fa')))

te_merged = np.concatenate( tesets, axis=0 )
tr_merged = np.concatenate( trsets, axis=0 )
crbm = CRBM.loadModel(outputdir + "/jund_joint.pkl")

trf = crbm.getHitProbs(tr_merged).sum(axis=(2,3))
tef = crbm.getHitProbs(te_merged).sum(axis=(2,3))

trl = getLabels(trsets)
tel = getLabels(tesets)

lrmodel = LogisticRegression(multi_class='multinomial', solver='newton-cg')
lrmodel.fit(trf, trl)
pred = lrmodel.predict(tef)

plotConfusionMatrix(cells, tel, pred,
        outputdir + "jund_celltype_discrimination_confmat.eps")

for i in range(len(crbm.getPFMs())):
    plotting.createSeqLogo(crbm.getPFMs()[i], 
            outputdir + "crbm_jund_{:d}.eps".format(i))

#per sequence TSNE
X = plotting.runTSNEPerSequence(crbm, te_merged)
lims = (X.min(axis=0)-1, X.max(axis=0)+1)

n = [tesets[0].shape[0] ]
for t in tesets[1:]:
    n.append(t.shape[0] + n[-1])

#test
Xsplit = np.split(X, n)
colors = cm.brg(np.linspace(0,1,len(Xsplit)))

Xscatter = {}
for i in range(len(cells)):
    Xscatter[cells[i]] = Xsplit[i]

plotting.tsneScatter(Xscatter, lims, colors, 
        outputdir + "tsne_jund_celltypes.pdf")

for cell, color in zip(Xscatter, colors):
    plotting.tsneScatter({cell:Xscatter[cell]}, lims, [color],
        outputdir + "tsne_jund_celltypes_{}.pdf".format(cell), legend = False)

plotting.plotTSNE_withpie(crbm, te_merged, X, lims,
        outputdir + "tsne_jund_motifcomposition.pdf")

lab = []
for i in range(len(cells)):
    lab += [ cells[i] ] * tesets[i].shape[0]

plotting.violinPlotMotifActivities(crbm, te_merged, lab, filename =\
        outputdir + "violin_jund_motifcomposition.eps")
