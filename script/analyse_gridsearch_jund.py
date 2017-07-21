import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

outputdir = os.environ["CRBM_OUTPUT_DIR"]

results = pd.read_csv(outputdir + "jund_gridsearch.csv", sep="\t", header=None)
results.columns = ["LearningRate", "Lambda", "Batchsize", "Rho", "SPM", 
        "Epochs", "Loss"]

fig, axarr = plt.subplots(2,2, figsize = (8,8))

for col, ax in zip(results.columns[:4], axarr.reshape((4))):
    #fig, ax = plt.figure(figsize=(7,4))
    results.boxplot(["Loss"], by= col, figsize=(7,4), ax = ax)
    ax.set_title("")
    fig = ax.get_figure()
    fig.suptitle('Cross-entropy loss')

fig.savefig(outputdir + "grid_jund.eps")


