import os
import joblib
import numpy as np
import sys
sys.path.append("../code")
from convRBM import CRBM
import plotting

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

# generate cRBM models
crbm_stem = CRBM(hyperParams=hyperParams)
crbm_fibro = CRBM(hyperParams=hyperParams)
crbm_merged = CRBM(hyperParams=hyperParams)

crbm_stem.loadModel(outputdir + "/stem_model.pkl")
crbm_fibro.loadModel(outputdir + "/fibro_model.pkl")
crbm_merged.loadModel(outputdir + "/merged_model.pkl")

training_stem, test_stem,training_fibro, test_fibro,training_merged,test_merged \
        = joblib.load(\
        outputdir + "dataset.pkl")

# evaluate free energy for testing data
print "Get free energy for both models..."
score_sc = crbm_stem.freeEnergy(test_merged)
score_fibro = crbm_fibro.freeEnergy(test_merged)
score = score_fibro - score_sc

labels = np.concatenate( (np.ones(test_stem.shape[0]), \
        np.zeros(test_fibro.shape[0])), axis=0 )

plotting.plotROC(labels, score, outputdir + "roc_oct4_mafk.eps")
plotting.plotPRC(labels, score, outputdir + "prc_oct4_mafk.eps")
crbm_stem.printHyperParams()

plt_stem = plotting.plotMotifsWithOccurrences(crbm_stem, test_stem, \
        outputdir + "oct4_model_features.eps")
plt_fibro = plotting.plotMotifsWithOccurrences(crbm_fibro, test_fibro, \
        outputdir + "mafk_model_features.eps")
plt_all = plotting.plotMotifsWithOccurrences(crbm_merged, test_merged, \
        outputdir + "joint_model_features.eps")

#per position TSNE
X = plotting.runTSNEPerPosition(crbm_merged, test_merged)

Xoct4, Xmafk = X[:test_stem.shape[0]], X[test_stem.shape[0]:]
color =labels

lims = (X.min(axis=0)-1, X.max(axis=0)+1)

plotting.tsneScatter({"Oct4":Xoct4, "Mafk":Xmafk}, lims,
            outputdir + "tsne_clustering_oct4_mafk_pp.pdf")

plotting.plotTSNE_withpie(crbm_merged, test_merged, X, lims, \
        outputdir + "tsne_clustering_pie_pp.pdf")

#per sequence TSNE
X = plotting.runTSNEPerSequence(crbm_merged, test_merged)
lims = (X.min(axis=0)-1, X.max(axis=0)+1)

Xoct4, Xmafk = X[:test_stem.shape[0]], X[test_stem.shape[0]:]

plotting.tsneScatter({"Oct4":Xoct4, "Mafk":Xmafk}, lims,
            outputdir + "tsne_clustering_oct4_mafk_ps.pdf")

plotting.plotTSNEPerSequence_withpie(crbm_merged, test_merged, X, lims,\
        outputdir + "tsne_clustering_pie_ps.pdf")

plotting.plotTSNEPerSequence_withpie(crbm_merged, test_stem, Xoct4, lims,\
        outputdir + "oct4_tsne_clustering_pie_ps.pdf")

plotting.plotTSNEPerSequence_withpie(crbm_merged, test_fibro, Xmafk, lims,\
        outputdir + "mafk_tsne_clustering_pie_ps.pdf")

lab = [ "Oct4"] * test_stem.shape[0] + [ "Mafk" ] * test_fibro.shape[0]
plotting.violinPlotMotifActivities(crbm_merged, test_merged, labels, filename =\
        outputdir + "violin_oct4_mafk_motif_occurrences.eps")
