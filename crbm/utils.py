import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import scipy
from weblogolib import LogoData, LogoFormat, LogoOptions, Alphabet
from weblogolib import classic, png_print_formatter, formatters
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from Bio.motifs.jaspar import write as write_motif
from Bio.motifs.jaspar import Motif
import os

def saveMotifs(model, path, name = "mot", fformat = 'jaspar'):
    """Save weight-matrices as PFMs.

    This method converts the cRBM weight-matrices to position frequency matrices
    and stores each matrix in a single file with ending .pfm.

    Parameters
    -----------
    model : :class:`CRBM` object
        A cRBM object.

    path : str
        Directory in which the PFMs should be stored.

    name : str
        File prefix for the motif files. Default: 'mot'.
    fformat : str
        File format of the motifs. Either 'jaspar' or 'tab'. Default: 'jaspar'.
    """
    
    pfms = model.getPFMs()
    alphabet = ['A','C','G','T']
    for i in range(len(pfms)):
        cnts = {}
        for c in range(len(alphabet)):
            cnts[alphabet[c]] = pfms[i][c]
        mot = Motif(name+str(i+1), name+str(i+1), counts = cnts)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "{}{:d}.{}".format(name, i, "pfm")), "w") as f:
            f.write(write_motif([mot], fformat))
        f.close()

def createSeqLogos(model, path, fformat = 'eps'):
    """Create sequence logos for all cRBM motifs

    Parameters
    -----------
    model : :class:`CRBM` object
        A cRBM object.
    path : str
        Output folder.
    fformat : str
        File format for storing the sequence logos. Default: 'eps'.
    """
    pfms = model.getPFMs()
    if not os.path.exists(path):
        os.makedirs(path)

    for idx in range(len(model.getPFMs())):
        pfm = pfms[idx]
        createSeqLogo(pfm, path + "/logo{:d}.{}".format(idx+1, fformat),
                fformat)

def createSeqLogo(pfm, filename, fformat = 'eps'):
    """Create sequence logo for an individual cRBM motif.

    Parameters
    -----------
    pfm : numpy-array
        2D numpy array representing a PFM. See :meth:`CRBM.getPFMs`
    path : str
        Output folder.
    fformat : str
        File format for storing the sequence logos. Default: 'eps'.
    """
    alph = Alphabet('ACGT')
    weblogoData = LogoData.from_counts(alph, pfm.T)#, c)#, learner.c.get_value().reshape(-1))
    weblogoOptions = LogoOptions(color_scheme=classic)
    weblogoFormat = LogoFormat(weblogoData, weblogoOptions)
    content = formatters[fformat](weblogoData, weblogoFormat)
    f = open(filename, "w")
    f.write(content)
    f.close()

def positionalDensityPlot(model, seqs, filename = None):
    """Positional enrichment of the motifs.

    This function creates a figure that illustrates a positional
    enrichment of all cRBM motifs in the given set of sequences.

    Parameters
    -----------
    model : :class:`CRBM` object
        A cRBM object
    seqs : numpy-array
        A set of DNA sequences in *one-hot* encoding.
        See :func:`crbm.sequences.seqToOneHot`
    filename : str
        Filename for storing the figure. If ``filename = None``,
        no figure will be stored.
    """

    # get motifs and hit observer

    pfms = model.getPFMs()

    h = model.motifHitProbs(seqs)

    #mean over sequences
    mh=h.mean(axis=(0,2))

    fig = plt.figure(figsize=(10,7))

    ax = fig.add_axes([.1,.1, .6,.75])
    colors = cm.rainbow(np.linspace(0,1,mh.shape[0]))
    for m_idx in range(mh.shape[0]):
        smh = np.convolve(mh[m_idx,:], 
                scipy.stats.norm.pdf(np.linspace(-3,3,num=10)), mode='same')
        plt.plot(range(mh.shape[1]), smh, color=colors[m_idx], 
                label='Motif {:d}'.format(m_idx +1))
    plt.xlabel("Position")
    plt.ylabel("Average motif match probability")
    plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def runTSNE(model, seqs):
    """Run t-SNE on the motif abundances.

    This function produces a clustering of the sequences
    using t-SNE based on the motif matches in the sequences.
    Accordingly, the sequences are projected onto a 2D hyper-plane
    in which similar sequences are located in close proximity.

    Parameters
    -----------
    model : :class:`CRBM` object
        A cRBM object
    seqs : numpy-array
        A set of DNA sequences in *one-hot* encoding.
        See :func:`crbm.sequences.seqToOneHot`
    """

    #hiddenprobs = model.fePerMotif(seqs)
    hiddenprobs = model.motifHitProbs(seqs)
    hiddenprobs = hiddenprobs.max(axis=(2,3))

    hreshaped = hiddenprobs.reshape((hiddenprobs.shape[0], 
        np.prod(hiddenprobs.shape[1:])))
    model = TSNE()
    return model.fit_transform(hreshaped)

def tsneScatter(data, lims = None, colors = None, filename = None, legend = True):
    """Scatter plot of t-SNE clustering.

    Parameters
    -----------
    data : dict
        Dictionary containing the dataset name (keys) and data itself (values).
        The data is assumed to have been generated using :func:`runTSNE`.

    lims : tuple
        Optional parameter containing the x- and y-limits for the figure.
        If None, the limits are automatically determined.
        For example: ``lims = ([xmin, ymin], [xmax, ymax])``

    colors : matplotlib.cm
        Optional colormap to illustrate the datapoints.
        If None, a default colormap will be used.

    filename : str
        Filename for storing the figure. Default: None, means 
        the figure stored but directly displayed.

    legend : bool
        Include the legend into the figure. Default: True
    """
        

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([.1,.1, .6,.75])

    if not colors:
        colors = cm.brg(np.linspace(0,1,len(data)))

    for name, color  in zip(data, colors):
        plt.scatter(x=data[name][:,0], y=data[name][:,1], 
                c=color, label = name, alpha=.3)
    if lims:
        plt.xlim(lims[0][0], lims[1][0])
        plt.ylim(lims[0][1], lims[1][1])

    if legend:
        plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
    plt.axis('off')

    if filename:
        fig.savefig(filename, dpi = 700)
    else:
        plt.show()

def tsneScatterWithPies(model, seqs, tsne, lims = None, filename= None):
    """Scatter plot of t-SNE clustering.

    This function produces a figure in which sequences are represented
    as pie chart in a 2D t-SNE hyper-plane obtained with :func:`runTSNE`.
    Moreover, the pie pieces correspond to the individual :class:`CRBM`
    motifs and the sizes represent the enrichment/abundance of the
    motifs in the respective sequences.

    Parameters
    -----------
    model : :class:`CRBM` object
        A cRBM object

    seqs : numpy-array
        DNA sequences represented in *one-hot* encoding.
        See :func:`crbm.sequences.seqToOneHot`.
    tsne : numpy-array
        2D numpy array representing the sequences projected onto
        the t-SNE hyper-plane that was obtained with :func:`runTSNE`.

    lims : tuple
        Optional parameter containing the x- and y-limits for the figure.
        For example:

        ([xmin, ymin], [xmax, ymax])

    filename : str
        Filename for storing the figure.
    """

    hiddenprobs = model.motifHitProbs(seqs)
    probs = hiddenprobs

    pmax=probs.max(axis=(0,2,3))
    pmedian=np.median(probs.max(axis=(2,3)), axis=(0))
    pcurrent = probs.max(axis = (2,3))

    colors = cm.rainbow(np.linspace(0,1,probs.shape[1]))

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_axes([.1,.1, .6,.75])
    #given a probability matrix (N x K x 1 x 200)
    for idx, col in zip(range(probs.shape[1]), colors):
        
        markx = [0] + np.cos([idx*2*np.pi/probs.shape[1], 2*np.pi*(idx+1)/probs.shape[1]]).tolist()
        marky = [0] + np.sin([idx*2*np.pi/probs.shape[1], 2*np.pi*(idx+1)/probs.shape[1]]).tolist()

        markxy = list(zip(markx, marky))

        #X, Y = tsne[:,0], tsne[:,1]
        
        s =150*(pcurrent[:, idx]-pmedian[idx])/(pmax[idx]-pmedian[idx])
        s[s<=0]=0.
        plt.scatter(tsne[:,0], tsne[:,1], marker=(markxy, 0),
                s=s, color = col, label = "Motif " + str(idx +1), alpha = .6)

    if lims:
        plt.xlim(lims[0][0], lims[1][0])
        plt.ylim(lims[0][1], lims[1][1])
    plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
    plt.axis('off')
    if filename:
        fig.savefig(filename, dpi=700)
    else:
        plt.show()

def violinPlotMotifMatches(model, data, filename = None):
    """Violin plot of motif abundances.

    This function summarized the relative motif abundances of
    the :class:`CRBM` motifs
    in a given set of sequences (e.g. sequences with different functions).

    Parameters
    -----------
    model : :class:`CRBM` object
        A cRBM object
    data : dict
        Dictionary with keys representing dataset-names
        and values a set of DNA sequences in *one-hot* encoding.
        See :func:`crbm.sequences.seqToOneHot`.
    filename : str
        Filename to store the figure. Default: None,
        the figure will be dirctly disployed.
    """

    seqs = np.concatenate(data.values(), axis=0)
    labels = []
    for k in data:
        labels += [k] * data[k].shape[0]

    hiddenprobs = model.motifHitProbs(seqs)
    probs = hiddenprobs.mean(axis=(2,3))

    probs = probs / np.amax(probs,axis=0, keepdims=True)

    fig = plt.figure(figsize = (8,5))
    df = pd.DataFrame(data=probs, columns = [ "Motif "+str(i +1) \
            for i in range(probs.shape[1])])
    df["TF"] = pd.DataFrame(data = pd.Series(labels, name="TF"))

    dfm = pd.melt(df, value_vars = df.columns[:-1], id_vars = "TF")
    g = sns.violinplot(x='variable', y='value', hue='TF', data=dfm,
            palette="Set2")

    g.set_xlabel("")
    g.set_ylabel("Normalized motif abundance")
    locs, labels = plt.xticks()
    #g.set_xticklabels(rotation=30)
    plt.setp(labels, rotation=30)
    #g.set_ylim(0,1)

    if filename:
        fig.savefig(filename, dpi=700)
    else:
        plt.show()

