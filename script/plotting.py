import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, auc
import scipy
import theano
import theano.tensor as T
import matplotlib as mpl
from matplotlib import gridspec
from weblogolib import LogoData, LogoFormat, LogoOptions, Alphabet
from weblogolib import classic, png_print_formatter, eps_formatter, png_formatter
from cStringIO import StringIO
import numpy as np
from sklearn.manifold import TSNE
from sklearn import metrics
import getData as dataRead
import pandas as pd
import seaborn as sns

mpl.rcParams['figure.facecolor'] = '0.75'
mpl.rcParams['grid.color'] = 'black'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

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

def plotTSNE_withpie_withmotif(model, seqs, tsne, lims, filename= None):

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

