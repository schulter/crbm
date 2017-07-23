#Convolutional restricted Boltzmann machine for learning DNA sequence features

This package contains functionality to extract redundant
DNA sequence features from a set of provided sequences.


# Tutorial example

The following example analysis demonstrates the functionality
of the package.
First, we load a set of example sequences and convert them
to one-hot encoding, which is required as input for the cRBM.
```
import pkg_resources

# Example fasta file of Oct4 ChIP-seq peaks

fafile = = pkg_resources.resource_filename('crbm','data/oct4.fa')

# Load a set of sample sequences from a
# fasta file

seqs = readSeqsFromFasta(fafile)


# Transform the sequences to one-hot encoding

onehot = seqToOneHot(seqs)
```

Next, we instantiate a cRBM to learn 10 motifs
of length 15 bp and train it on the provided sequences.

```
import crbm

model = crbm.CRBM(num_motifs = 10, motif_length = 15)

# Fit the model
model.fit(seqs)
```

After having trained the model, the parameters of the
cRBM can be stored and reused later to instantiate it.
This will not only store the parameters, but also all
hyperparameters for the model.

```
# Save params
model.saveModel('oct4_model_params.pkl')

# Reinstantiate model
model = crbm.CRBM.loadModel('oct4_model_params.pkl')
```

The parameters of the cRBM can be converted to
a position frequency matrix (PFM) representation
by which can readily be depicted as sequence logos
```
model.getPFMs()

# store the motifs as sequence logos

import crbm.utils

crbm.utils.createSeqlogos(crbm.getPFMs(), "logos/")
```

One can also investigate the localization of
each motif in a profile across multiple sequences

```
# Per-position motif match probabilities

matches = model.getMatchProbs(one_hot[:100])
```
where `matches` is a numpy array of dimensions
`Nseqs x num_motifs x 1 x Seqlengths - motif_length + 1`.

A positional profile of the matches across the sequences
can be drawn like
```
positionalDensityPlot(crbm, one_hot[:100], filename = 'densityplot.png')
```
