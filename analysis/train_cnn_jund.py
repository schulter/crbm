import matplotlib.pyplot as plt
import numpy as np
import random
import time
from datetime import datetime
from sklearn import metrics
import joblib
import os
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten


from getData import seqToOneHot, readSeqsFromFasta

outputdir = os.environ["CRBM_OUTPUT_DIR"]

# get the data
cells = ["GM12878", "H1hesc", "Hela", "HepG2", "K562"]
trsets = []
tesets = []

for cell in cells:
    trsets.append(seqToOneHot(readSeqsFromFasta(
    '../data/jund_data/' + cell + '_only_train.fa')))
    tesets.append(seqToOneHot(readSeqsFromFasta(
    '../data/jund_data/' + cell + '_only_test.fa')))


te_merged = np.concatenate( tesets, axis=0 )
tr_merged = np.concatenate( trsets, axis=0 )

def getLabels(dataset):
    l = np.concatenate([ np.repeat(i, dataset[i].shape[0]) for i in range(len(dataset)) ])
    eye = np.eye(len(dataset))
    oh = np.asarray([ eye[:,i] for i in l])
    return oh

troh = getLabels(trsets)
teoh = getLabels(tesets)

model = Sequential()
model.add(Conv2D(20, kernel_size = (4, 15), input_shape = (1,4,200),
    activation = 'sigmoid'))
model.add(MaxPooling2D(pool_size = (1,200 - 15 + 1)))
model.add(Flatten())
model.add(Dense(5, activation='sigmoid'))

model.compile(optimizer = 'adadelta', loss='categorical_crossentropy',
    metrics=['accuracy'])

model.fit(tr_merged, troh, epochs=100, batch_size = 20)

pred = []
for te in tesets:
    pred.append(model.predict(te))

def plotDiscriminativePerformance(cells, pred, fct, name, filename=None):
    scores = np.ones((len(pred),len(pred)))
    for i in range(len(cells) - 1):
        for j in range(i+1, len(cells)):
            pred_merged = np.concatenate( (pred[i], pred[j]), axis=0 )
            telab = np.concatenate( (np.ones(pred[i].shape[0]), \
                    np.zeros(pred[j].shape[0])), axis=0 )
            score_i = np.log(pred_merged[:,i])
            score_j = np.log(pred_merged[:,j])
            score  = score_i - score_j
            scores[j,i] = fct(telab,score)
    df = pd.DataFrame(scores, columns = cells, index = cells)
    import seaborn as sns
    sns.set(style="white")
    # Generate a mask for the upper triangle
    mask = np.zeros_like(scores, dtype=np.bool)
    mask[np.tril_indices_from(mask)] = True
    vmin = scores[mask].min()
    vmax = scores[mask].max()
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(df, ax=ax, mask=mask==False, cmap=cmap, vmin = vmin, vmax=vmax, 
            square=True, annot= True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_title(name)
    if filename:
        f.savefig(filename, dpi = 1000)
    else:
        plt.show()

plotDiscriminativePerformance(cells, pred, metrics.average_precision_score, "auPRC")
plotDiscriminativePerformance(cells, pred, metrics.roc_auc_score, "auROC")
