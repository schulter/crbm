import os
import joblib
import numpy as np
import sys
sys.path.append("../code")
from convRBM import CRBM
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from getData import seqToOneHot, readSeqsFromFasta
from Bio.motifs.matrix import PositionWeightMatrix
from Bio import motifs
from Bio.Alphabet import IUPAC

import plotting

outputdir = os.environ["CRBM_OUTPUT_DIR"]

# generate cRBM models
crbm_oct4 = CRBM.loadModel(outputdir + "/oct4_model.pkl")
crbm_mafk = CRBM.loadModel(outputdir + "/mafk_model.pkl")
crbm_merged = CRBM.loadModel(outputdir + "/oct4_mafk_joint_model.pkl")

tr_o = seqToOneHot(readSeqsFromFasta('../data/stemcells_train.fa'))
te_o = seqToOneHot(readSeqsFromFasta('../data/stemcells_test.fa'))
tr_m = seqToOneHot(readSeqsFromFasta('../data/fibroblast_train.fa'))
te_m = seqToOneHot(readSeqsFromFasta('../data/fibroblast_test.fa'))

te_merged = np.concatenate( (te_o, te_m), axis=0 )
tr_merged = np.concatenate( (tr_o, tr_m), axis=0 )
labels = np.concatenate( (np.ones(te_o.shape[0]),
    np.zeros(te_m.shape[0])), axis=0 )


# motif 8 corresponds to Oct4
for i in range(len(crbm_oct4.getPFMs())):
    plotting.createSeqLogo(crbm_oct4.getPFMs()[i], 
            outputdir + "crbm_oct4_{:d}.eps".format(i))

for i in range(len(crbm_mafk.getPFMs())):
    plotting.createSeqLogo(crbm_mafk.getPFMs()[i], 
            outputdir + "crbm_mafk_{:d}.eps".format(i))

for i in range(len(crbm_merged.getPFMs())):
    plotting.createSeqLogo(crbm_merged.getPFMs()[i], 
            outputdir + "crbm_joint_oct4_mafk_{:d}.eps".format(i))


plotting.positionalDensityPlot(crbm_oct4, te_o, 
       outputdir + "oct4_positional_profile.eps")
plotting.positionalDensityPlot(crbm_mafk, te_m, 
       outputdir + "mafk_positional_profile.eps")
plotting.positionalDensityPlot(crbm_merged, te_merged, 
       outputdir + "joint_oct4_mafk_positional_profile.eps")

# 
plt_oct4 = plotting.plotMotifsWithOccurrences(crbm_oct4, te_o, \
        outputdir + "oct4_model_features.eps")
plt_mafk = plotting.plotMotifsWithOccurrences(crbm_mafk, te_m, \
        outputdir + "mafk_model_features.eps")
plt_all = plotting.plotMotifsWithOccurrences(crbm_merged, te_merged, \
        outputdir + "joint_model_features.eps")

#per sequence TSNE
X = plotting.runTSNEPerSequence(crbm_merged, te_merged)
lims = (X.min(axis=0)-1, X.max(axis=0)+1)

Xoct4, Xmafk = X[:te_o.shape[0]], X[te_o.shape[0]:]

colors = cm.brg(np.linspace(0,1,2))

plotting.tsneScatter({"Oct4":Xoct4, "Mafk":Xmafk}, lims, colors,
            outputdir + "tsne_clustering_oct4_mafk_ps.pdf")

plotting.plotTSNE_withpie(crbm_merged, te_merged, X, lims,\
        outputdir + "tsne_clustering_pie_ps.pdf")

plotting.plotTSNE_withpie(crbm_merged, te_o, Xoct4, lims,\
        outputdir + "oct4_tsne_clustering_pie_ps.pdf")

plotting.plotTSNE_withpie(crbm_merged, te_m, Xmafk, lims,\
        outputdir + "mafk_tsne_clustering_pie_ps.pdf")

lab = [ "Oct4"] * te_o.shape[0] + [ "Mafk" ] * te_m.shape[0]
plotting.violinPlotMotifActivities(crbm_merged, te_merged, lab, filename =
        outputdir + "violin_oct4_mafk_motif_occurrences.eps")
