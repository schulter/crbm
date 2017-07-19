import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

outputdir = os.environ["CRBM_OUTPUT_DIR"]

results = pd.read_csv(outputdir + "gridsearch_om_v2.csv", sep="\t", header=None)
results.columns = ["LearningRate", "Lambda", "Batchsize", "Rho", "SPM", 
        "Epochs", "auROC","auPRC"]

for col in results.columns[:4]:
    #fig, ax = plt.figure(figsize=(7,4))
    ax = results.boxplot(["auROC","auPRC"], by= col, figsize=(7,4))
    #ax = results.boxplot(["auROC","auPRC"], by= ["Lambda", "SPM"], figsize=(7,4))
    fig = ax[0].get_figure()
    fig.suptitle('')
    fig.savefig(outputdir + "grid_{}_om.eps".format(col))


