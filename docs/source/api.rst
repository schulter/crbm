===
API
===

    
        
Convolutional restricted Boltzmann machine
==========================================

:class:`CRBM` contains the main functionality of the package
for training, evaluating and investigating the model.

.. currentmodule:: crbm

.. autosummary:: 

    CRBM.fit
    CRBM.freeEnergy
    CRBM.motifHitProbs
    CRBM.getPFMs
    CRBM.saveModel
    CRBM.loadModel


.. automodule:: crbm

.. autoclass:: CRBM
    :members:

Sequence-related utilies
===========================

Functions for loading DNA sequences in fasta format
and for convert them to the required *one-hot* encoding.

.. currentmodule:: crbm.sequences

.. autosummary::

    readSeqsFromFasta
    seqToOneHot
    splitTrainingTest
    
.. automodule:: crbm.sequences
    :members: readSeqsFromFasta, seqToOneHot, splitTrainingTest

Utils
=======

This part presents functions contained in :mod:`crbm.utils` that
support investigating the results of the :class:`cRBM`,
including by generating position frequency matrices, sequence logos
and clustering plots.

.. currentmodule:: crbm.utils

.. autosummary::

    saveMotifs
    createSeqLogos
    createSeqLogo
    positionalDensityPlot
    runTSNE
    tsneScatter
    tsneScatterWithPies
    violinPlotMotifMatches

.. automodule:: crbm.utils
    :members: saveMotifs, positionalDensityPlot, runTSNE, tsneScatter,
            createSeqLogos, createSeqLogo,
            tsneScatterWithPies, violinPlotMotifMatches

Sample dataset
===============

The package contains a small sample dataset consisting
of *Oct4* ChIP-seq sequences of embryonic stem cells from
ENCODE [1]_.

.. autofunction:: crbm.sequences.load_sample


.. [1] ENCODE Project Consortium and others. (2012).
        An integrated encyclopedia of DNA elements in the human genome. Nature.
