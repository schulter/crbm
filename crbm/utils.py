import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
from weblogolib import LogoData, LogoFormat, LogoOptions, Alphabet
from weblogolib import classic, png_print_formatter, formatters
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns


def createSeqLogos(crbm, path, fformat = 'eps'):
    pfms = crbm.getPFMs()

    for idx in range(len(crbm.getPFMs)):
        pfm = pfms[i]
        createSeqLogo(pfm, path + "/logo{:d}.{}".format(idx+1, fformat),
                fformat)

def createSeqLogo(pfm, filename, fformat = 'eps'):
    alph = Alphabet('ACGT')
    weblogoData = LogoData.from_counts(alph, pfm.T)#, c)#, learner.c.get_value().reshape(-1))
    weblogoOptions = LogoOptions(color_scheme=classic)
    weblogoFormat = LogoFormat(weblogoData, weblogoOptions)
    content = formatters[fformat](weblogoData, weblogoFormat)
    f = open(filename, "w")
    f.write(content)
    f.close()

def positionalDensityPlot(crbm, seqs, filename = None):
    # get motifs and hit observer

    pfms = crbm.getPFMs()

    h = crbm.getHitProbs(seqs)

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
   # plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def runTSNE(model, seqs):

    #hiddenprobs = model.fePerMotif(seqs)
    hiddenprobs = model.getHitProbs(seqs)
    hiddenprobs = hiddenprobs.max(axis=(2,3))

    hreshaped = hiddenprobs.reshape((hiddenprobs.shape[0], 
        np.prod(hiddenprobs.shape[1:])))
    model = TSNE()
    return model.fit_transform(hreshaped)

def tsneScatter(data, lims, colors, filename = None, legend = True):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([.1,.1, .6,.75])
    for name, color  in zip(data, colors):
        plt.scatter(x=data[name][:,0], y=data[name][:,1], 
                c=color, label = name, alpha=.3)
    plt.xlim(lims[0][0], lims[1][0])
    plt.ylim(lims[0][1], lims[1][1])
    if legend:
        plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
    plt.axis('off')
    if filename:
        print(filename)
        fig.savefig(filename, dpi = 700)
    else:
        plt.show()

def tsneScatterWithPies(model, seqs, tsne, lims, filename= None):

    hiddenprobs = model.getHitProbs(seqs)
    probs = hiddenprobs

    pmin=probs.max(axis=(2,3)).min(axis=0)
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

    plt.xlim(lims[0][0], lims[1][0])
    plt.ylim(lims[0][1], lims[1][1])
    plt.legend(bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
    plt.axis('off')
    if filename:
        fig.savefig(filename, dpi=700)
    else:
        plt.show()

def violinPlotMotifMatches(model, seqs, labels, filename = None):
    hiddenprobs = model.getHitProbs(seqs)
    probs = hiddenprobs.mean(axis=(2,3))

    probs = probs / np.amax(probs,axis=0, keepdims=True)

    fig = plt.figure(figsize = (8,5))
    df = pd.DataFrame(data=probs, columns = [ "Motif "+str(i +1) \
            for i in range(probs.shape[1])])
    df["TF"] = pd.DataFrame(data = pd.Series(labels, name="TF"))

    dfm = pd.melt(df, value_vars = df.columns[:-1], id_vars = "TF")
    g = sns.violinplot(x='variable', y='value', hue='TF', data=dfm,
            palette="Set2")

    g.set_xlabel("Motifs")
    g.set_ylabel("Normalized motif match enrichment")
    locs, labels = plt.xticks()
    #g.set_xticklabels(rotation=30)
    plt.setp(labels, rotation=30)
    #g.set_ylim(0,1)

    if filename:
        fig.savefig(filename, dpi=700)
    else:
        plt.show()

