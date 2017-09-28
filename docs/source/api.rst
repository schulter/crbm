===
SECOMO API
===

    
        
Convolutional restricted Boltzmann machine (cRBM)
==========================================

:class:`CRBM` contains the main functionality of SECOMO
for training, evaluating and investigating models.

.. currentmodule:: secomo

.. autosummary:: 

    CRBM.fit
    CRBM.freeEnergy
    CRBM.motifHitProbs
    CRBM.getPFMs
    CRBM.saveModel
    CRBM.loadModel


.. automodule:: secomo

.. autoclass:: CRBM
    :members:

Sequence-related utilies
===========================

Functions for loading DNA sequences in fasta format
and for convert them to the required *one-hot* encoding.

.. currentmodule:: secomo.sequences

.. autosummary::

    readSeqsFromFasta
    seqToOneHot
    splitTrainingTest
    
.. automodule:: secomo.sequences
    :members: readSeqsFromFasta, seqToOneHot, splitTrainingTest

Utils
=======

This part presents functions contained in :mod:`secomo.utils` that
help you investigate the results of a trained SECOMO model.
It features generating position frequency matrices, sequence logos
and clustering plots.

.. currentmodule:: secomo.utils

.. autosummary::

    saveMotifs
    createSeqLogos
    createSeqLogo
    positionalDensityPlot
    runTSNE
    tsneScatter
    tsneScatterWithPies
    violinPlotMotifMatches

.. automodule:: secomo.utils
    :members: saveMotifs, positionalDensityPlot, runTSNE, tsneScatter,
            createSeqLogos, createSeqLogo,
            tsneScatterWithPies, violinPlotMotifMatches

Sample dataset
===============

The package contains a small sample dataset consisting
of *Oct4* ChIP-seq sequences of embryonic stem cells from
ENCODE [1]_.

.. autofunction:: secomo.sequences.load_sample


.. [1] ENCODE Project Consortium and others. (2012).
        An integrated encyclopedia of DNA elements in the human genome. Nature.
