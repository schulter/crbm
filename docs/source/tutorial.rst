========
Tutorial
========

This section is intended to demonstrate the main functionality of
SECOMO on a small toy dataset.

Loading sample dataset
----------------------

Assuming you have successfully install the :mod:`secomo` package,
you can load a sample dataset consisting of *Oct4* ChIP-seq peaks
from embryonic stem cells (ESCs)

.. code-block:: python

    import secomo

    # Obtain sample sequences in one-hot encoding
    onehot = secomo.load_sample()

The original DNA sequences first need to be converted to *one-hot* encoding

.. note::

    The sample sequences are contained in the :mod:`secomo` package in
    fasta format. Generally, fasta files can be loaded and converted 
    to *one-hot* encoding according to

    .. code-block:: python

        # Convert to one-hot
        seqs = secomo.readSeqsFromFasta("path/to/seq.fa")
        onehot = secomo.seqToOneHot(seqs)

Train SECOMO's convolutional RBM model
----------

Next, we instantiate a cRBM to learn 10 motifs
of length 15 bp and train it on the provided sequences

.. code-block:: python

    # Obtain a cRBM object
    model = secomo.CRBM(num_motifs = 10, motif_length = 15)

    # Fit the model
    model.fit(onehot)

.. note::

    The `CRBM` object can be instantiated with a number of training-related
    hyper-parameters, e.g. number of epochs. See API for more information.

.. note::

    Optionally, the `fit` method also accepts a validation dataset
    on which the training progress is monitored. This makes it easier
    to detect overfitting. If no validation set is supplied,
    the progress is reported on the training set.


Save and restore the parameters
+++++++++++++++++++++++++++++++

After having trained the model, 
it is common to store the parameters
reused them later for a subsequent analysis.
To this end, the following methods can be invoked

.. code-block:: python

    # Save parameters and hyper-parameters
    model.saveModel('oct4_model_params.pkl')

    # Reinstantiate model
    model = secomo.CRBM.loadModel('oct4_model_params.pkl')

Position frequency matrices
---------------------------

A common way to investigate patterns in DNA sequences is
to model them as *position frequency matrices* (PFMs) which can be
then nicely visualized.
The model parameters (e.g. the weight matrices) learned by the
cRBM can be converted to such PFMs,
which can then be used for further downstream analysis.
For this purpose one can utilize

.. code-block:: python

    # Get a list of numpy matrices representing PFMs
    model.getPFMs()

    # Store the PFMs (by default in 'jaspar' format)
    # in the folder './pfms/'
    secomo.saveMotifs(model, path = './pfms/')

PFMs are frequently visualized in terms of sequence logos
which can be obtained by

.. code-block:: python

    # Writes all logos in the logos/ directory
    secomo.utils.createSeqLogos(model, path = "./logos/")

    # Alternatively, an individual sequence logo can be created:
    # Get first motif
    pfm = model.getPFMs()[0]

    # Create a corresponding sequence logo
    secomo.utils.createSeqLogo(pfm, filename = "logo1.png", fformat = "png")


Motif matches
-------------

Next, we inspect at which positions in a set of DNA sequences
motif matches are present.
The per-position motif match probabilities can be obtained as follows

.. code-block:: python

    # Per-position motif match probabilities
    # for the first 100 sequences
    matches = model.motifHitProbs(onehot[:100])

Here, ``matches`` represents a 4D numpy array comprising the match
probabilities with dimensions
`Nseqs x num_motifs x 1 x (Seqlengths - motif_length + 1)`.

An average profile of match probabilities per-position
can be illustrated using

.. code-block:: python

    # Plot positional enrichment for all motifs in the given
    # test sequences
    secomo.positionalDensityPlot(model, onehot[:100], filename = './densityplot.png')


Clustering analysis
-------------------

Finally, we shall demonstrate that the sequence activations of similar sequences tend to
cluster together. In order to visualize the clusters, we run TSNE dimensionality reduction using

.. code-block:: python

    # Run t-SNE dim reduction to 2D
    tsne = secomo.runTSNE(model, onehot)

    # Visualize the results in a scatter plot
    secomo.tsneScatter({'Oct4': tsne}, filename = './tsnescatter.png')

    # Visualize the results in the scatter plot
    # by augmenting with the respective motif abundances
    secomo.tsneScatterWithPies(model, onehot, tsne, filename = "./tsnescatter_pies.png")

Motif enrichment across different sets of sequences
---------------------------------------------------

This part concerns the analysis of multiple datasets
with the same cRBM.
In order to find out whether a specific 
motif (e.g. weight matrices)
is enriched or depleted in a certain dataset relative
to the others a violin plot can be created.
In the following example, we just artificially split
the *Oct4* dataset into *set1* and *set2* to illustrate
the function

.. code-block:: python

    # Assemble multiple datasets as follows
    data = {'set1': onehot[:1500], 'set2': onehot[1500:]}

    secomo.violinPlotMotifMatches(model, data, 
            filename = os.path.join(path, 'violinplot.png'))

    
Summary of the full analysis
----------------------------

The full tutorial code can be found in the Github repository: 
`crbm/tutorial.py <https://github.com/schulter/crbm/blob/master/crbm/tutorial.py>`_
