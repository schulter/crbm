import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

outputdir = os.environ["CRBM_OUTPUT_DIR"]

results = pd.read_csv(outputdir + "om_gridsearch.csv", sep="\t", header=None)
results.columns = ["LearningRate", "Lambda", "Batchsize", "Rho", "SPM", 
        "Epochs", "auROC"]

fig, axarr = plt.subplots(2,2, figsize = (8,8))
for col, ax in zip(results.columns[:4], axarr.reshape((4))):
    #fig, ax = plt.figure(figsize=(7,4))
    results.boxplot(["auROC"], by= col, figsize=(7,4), ax = ax)
    #ax = results.boxplot(["auROC","auPRC"], by= ["Lambda", "SPM"], figsize=(7,4))
    fig = ax.get_figure()
    ax.set_title("")
    fig.suptitle('auROC')


fig.savefig(outputdir + "grid_om.eps")
