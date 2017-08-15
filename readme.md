# Convolutional restricted Boltzmann machine for learning DNA sequence features

[![Build Status](https://travis-ci.org/wkopp/crbm.svg?branch=master)](https://travis-ci.org/wkopp/crbm)
[![Documentation Status](https://readthedocs.org/projects/crbm/badge/?version=latest)](http://crbm.readthedocs.io/en/latest/?badge=latest)

This package contains functionality to extract redundant
DNA sequence features from a set of provided sequences.
The package also contains utilities for plotting the revealed
motifs and conducting clustering analysis based
on the revealed motifs.

## System requirements

Minimal requirements:
`theano`, `numpy`, `scipy`, `joblib`, `Biopython`

Additional optional packages for explorative analysis:
`matplotlib`, `pandas`, `weblogolib`, `scikit-learn`, `seaborn`

Furthermore, we recommend to install `cuda`, which can be taken
advantage of by `theano`. See `theano` documention for installation instructions.


## Tutorial

The following example analysis demonstrates the functionality
of the package.
First, we load a set of example sequences and convert them
to one-hot encoding, which is required as input for the cRBM.
```python
import os
import crbm

# Example fasta file of Oct4 ChIP-seq peaks

fafile = os.path.join(crbm.__path__[0], '..','seq', 'oct4.fa')

# Load a set of sample sequences from a
# fasta file

seqs = crbm.readSeqsFromFasta(fafile)


# Transform the sequences to one-hot encoding

onehot = crbm.seqToOneHot(seqs)
```

Next, we instantiate a cRBM to learn 10 motifs
of length 15 bp and train it on the provided sequences.

```python
model = crbm.CRBM(num_motifs = 10, motif_length = 15, epochs = 30)

# Fit the model
model.fit(onehot[:3000], onehot[3000:])
```

After having trained the model, the parameters of the
cRBM can be stored and reused later to instantiate it.
This will not only store the parameters, but also all
hyperparameters for the model.

```python
# Save params
model.saveModel('oct4_model_params.pkl')

# Reinstantiate model
model = crbm.CRBM.loadModel('oct4_model_params.pkl')
```

The parameters of the cRBM can be converted to
a position frequency matrix (PFM) representation
by which can readily be depicted as sequence logos
```python
model.getPFMs()

# writes all logos in the logos/ directory
crbm.utils.createSeqLogos(model, "logos/")

# alternatively, a single logo can be stored 
# in a chosen format
crbm.utils.createSeqLogo(model.getPFMs()[0], "logo1.png", "png")
```


The positions at which motif matches occur
may be further investigated using
```python
# Per-position motif match probabilities

matches = model.getHitProbs(onehot[:100])
```
where `matches` is a numpy array of dimensions
`Nseqs x num_motifs x 1 x Seqlengths - motif_length + 1`.

A positional profile of the matches across the sequences
can be drawn like
```python
crbm.positionalDensityPlot(model, onehot[:100], filename = 'densityplot.png')
```

Finally, we shall demonstrate how to perform a clustering analysis
of the sequences under study based on the cRBM motifs.
To that end, we first run TSNE clustering using

```python
tsne = crbm.runTSNE(model, onehot)
```

```python
crbm.tsneScatterWithPies(model, onehot, tsne, filename = "tsnescatter.png")
```
