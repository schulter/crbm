import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, auc
import theano
import theano.tensor as T
from matplotlib import gridspec
from weblogolib import LogoData, LogoFormat, LogoOptions, Alphabet
from weblogolib import classic, png_print_formatter, png_formatter
from cStringIO import StringIO
import numpy as np
from sklearn.manifold import TSNE
from sklearn import metrics
import getData as dataRead
import pandas as pd
import seaborn as sns

def createWeblogo(pwm, highRes=False):
    alph = Alphabet('ACGT')
    print pwm.shape
    weblogoData = LogoData.from_counts(alph, pwm.T)#, c)#, learner.c.get_value().reshape(-1))
    weblogoOptions = LogoOptions(color_scheme=classic)
    weblogoFormat = LogoFormat(weblogoData, weblogoOptions)
    if highRes:
        x = png_print_formatter(weblogoData, weblogoFormat)
    else:
        x = png_formatter(weblogoData, weblogoFormat)
    fake_file = StringIO(x)
    return plt.imread(fake_file)


def plotMotifsWithOccurrences(crbm, seqs, filename = None):
    # get motifs and hit observer

    pfms = crbm.getPFMs()

    h = crbm.getHitProbs(seqs)

    #mean over sequences
    mh=h.mean(axis=(0,2))

    mean_occurrences = mh.mean(axis=1)
    sorted_motifs = mean_occurrences.argsort()[::-1]

    fig = plt.figure(figsize=(20, 2*len(pfms)))
    gs = gridspec.GridSpec(len(pfms), 2, width_ratios=[1, 1])
    plot_idx = 0

    for m_idx in sorted_motifs:
        # plot logo
        ax1 = plt.subplot(gs[plot_idx])
        logo = createWeblogo(pfms[m_idx])
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        plt.imshow(logo)
        plt.title('Sequence Logo {}'.format(m_idx))
        plot_idx += 1
        ax2 = plt.subplot(gs[plot_idx])
        ax2.get_xaxis().set_visible(False)
        plt.bar(range(mh.shape[1]), mh[m_idx])
        plt.title('Average motif match probability {}'.format(m_idx))
        plt.xlabel("Position")
        plot_idx += 1

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def runTSNEPerSequence(model, seqs):

    #hiddenprobs = model.fePerMotif(seqs)
    hiddenprobs = model.getHitProbs(seqs)
    hiddenprobs = hiddenprobs.max(axis=(2,3))

    hreshaped = hiddenprobs.reshape((hiddenprobs.shape[0], 
        np.prod(hiddenprobs.shape[1:])))
    model = TSNE()
    return model.fit_transform(hreshaped)

def tsneScatter(data, lims, filename = None):

    fig = plt.figure(figsize=(6, 6))

    colors = cm.brg(np.linspace(0,1,len(data)))

    for name, color  in zip(data, colors):
        plt.scatter(x=data[name][:,0], y=data[name][:,1], \
                c=color, label = name, alpha=.3)

    plt.xlim(lims[0][0], lims[1][0])
    plt.ylim(lims[0][1], lims[1][1])

    plt.legend(loc="lower right")
    plt.axis('off')
    if filename:
        fig.savefig(filename, dpi = 700)
    else:
        plt.show()

def plotTSNE_withpie(model, seqs, tsne, lims, filename= None):

    hiddenprobs = model.getHitProbs(seqs)
    probs = hiddenprobs

    pmin=probs.max(axis=(2,3)).min(axis=0)
    pmax=probs.max(axis=(0,2,3))
    pmedian=np.median(probs.max(axis=(2,3)), axis=(0))
    pcurrent = probs.max(axis = (2,3))

    colors = cm.rainbow(np.linspace(0,1,probs.shape[1]))

    fig = plt.figure(figsize = (6, 6))
    #given a probability matrix (N x K x 1 x 200)
    for idx, col in zip(range(probs.shape[1]), colors):
        
        markx = [0] + np.cos([idx*2*np.pi/probs.shape[1], 2*np.pi*(idx+1)/probs.shape[1]]).tolist()
        marky = [0] + np.sin([idx*2*np.pi/probs.shape[1], 2*np.pi*(idx+1)/probs.shape[1]]).tolist()

        markxy = list(zip(markx, marky))

        #X, Y = tsne[:,0], tsne[:,1]
        
        s =150*(pcurrent[:, idx]-pmedian[idx])/(pmax[idx]-pmedian[idx])
        s[s<=0]=0.
        plt.scatter(tsne[:,0], tsne[:,1], marker=(markxy, 0),
                s=s, color = col, label = "Motif " + str(idx), alpha = .6)

    plt.xlim(lims[0][0], lims[1][0])
    plt.ylim(lims[0][1], lims[1][1])
    plt.legend(loc="lower right")
    plt.axis('off')
    if filename:
        plt.savefig(filename, dpi=700)
    else:
        plt.show()

def plotTSNEPerSequence_withpie(model, seqs, tsne, lims, filename= None):

    hiddenprobs = model.getHitProbs(seqs)
    probs = hiddenprobs

    pmin=probs.max(axis=(2,3)).min(axis=0)
    pmax=probs.max(axis=(0,2,3))
    pmedian=np.median(probs.max(axis=(2,3)), axis=(0))
    pcurrent = probs.max(axis = (2,3))

    colors = cm.rainbow(np.linspace(0,1,probs.shape[1]))
    fig = plt.figure(figsize = (6, 6))

    pfms = model.getPFMs()
    gs = gridspec.GridSpec(len(pfms), 2, width_ratios=[.2, 1])
    plot_idx = 0

    for m_idx in range(len(pfms)):
        # plot logo
        ax1 = plt.subplot(gs[m_idx,0])
        logo = createWeblogo(pfms[m_idx])
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        plt.imshow(logo)
        plt.title('Motif {}'.format(m_idx), fontsize = 6)
        plot_idx += 1

    #given a probability matrix (N x K x 1 x 200)
    ax1 = plt.subplot(gs[:,1])
    for idx, col in zip(range(probs.shape[1]), colors):
        
        markx = [0] + np.cos([idx*2*np.pi/probs.shape[1], 2*np.pi*(idx+1)/probs.shape[1]]).tolist()
        marky = [0] + np.sin([idx*2*np.pi/probs.shape[1], 2*np.pi*(idx+1)/probs.shape[1]]).tolist()

        markxy = list(zip(markx, marky))

        #X, Y = tsne[:,0], tsne[:,1]
        
        s =150*(pcurrent[:, idx]-pmedian[idx])/(pmax[idx]-pmedian[idx])
        s[s<=0]=0.
        ax1.scatter(tsne[:,0], tsne[:,1], marker=(markxy, 0),
                s=s, color = col, label = "Motif " + str(idx), alpha = .6)

    plt.xlim(lims[0][0], lims[1][0])
    plt.ylim(lims[0][1], lims[1][1])
    ax1.legend(loc="lower right", fontsize = 6)
    ax1.axis('off')
    if filename:
        fig.savefig(filename, dpi=700)
    else:
        plt.show()

def violinPlotMotifActivities(model, seqs, labels, filename = None):
    hiddenprobs = model.getHitProbs(seqs)
    probs = hiddenprobs.mean(axis=(2,3))

    probs = probs / np.amax(probs,axis=0, keepdims=True)

    fig = plt.figure(figsize = (8,5))
    df = pd.DataFrame(data=probs, columns = [ "Motif "+str(i) \
            for i in range(probs.shape[1])])
    df["TF"] = pd.DataFrame(data = pd.Series(labels, name="TF"))

    dfm = pd.melt(df, value_vars = df.columns[:-1], id_vars = "TF")
    g = sns.violinplot(x='variable', y='value', hue='TF', data=dfm,
            palette="Set2")

    g.set_xlabel("Motifs")
    g.set_ylabel("Normalized motif match enrichment")
    #g.set_ylim(0,1)

    if filename:
        fig.savefig(filename, dpi=700)
    else:
        plt.show()

