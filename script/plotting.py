import matplotlib.pyplot as plt
import trainingObserver as observer
from sklearn.metrics import roc_curve, auc

def plotFE(model):
    # save trained model to file

		# plot
		plt.subplot(2, 1, 1)
		plt.ylabel('Free energy')
		title = "%f lr %d kmers %d numOfMotifs %d cd_k" % (model.hyper_params['learning_rate'],
																												model.hyper_params['motif_length'],
																												model.hyper_params['number_of_motifs'],
																												model.hyper_params['cd_k'])
		plt.title(title)

		plt.plot(observer.getObserver(model, "FE-test").scores)
		plt.plot(observer.getObserver(model, "FE-training").scores)

		plt.subplot(2, 1, 2)
		plt.ylabel('Reconstruction rate')
		plt.xlabel('Number Of Epoch')

		plt.plot(observer.getObserver(model, "Recon-test").scores)
		plt.plot(observer.getObserver(model, "Recon-training").scores)

		# save plot to file
		file_name_plot = "./errorPlot.png"
		plt.savefig(file_name_plot)


def plotROC(scores, texts, labels, filename):
    assert len(scores) == len(texts) #== len(labels)
    fig = plt.figure(figsize=(14, 8))

    for i in range(len(scores)):
        fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=scores[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=texts[i] + '(AUC = %0.2f)' % roc_auc)
    
    # plot the random line
    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    # set general parameters for plotting and so on
    plt.ylim([0.0, 1.05])
    plt.xlim([-0.05, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics for sparsity constraint')
    plt.legend(loc="lower right")
    fig.savefig(filename, dpi=400)

