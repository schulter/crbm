===
API
===

    
        
    
Convolutional restricted Boltzmann machine
==========================================

.. autosummary:: 

    crbm.CRBM.fit
    crbm.CRBM.freeEnergy
    crbm.CRBM.motifHitProbs
    crbm.CRBM.getPFMs
    crbm.CRBM.saveModel
    crbm.CRBM.loadModel

:class:`CRBM` contains the main functionality of the package
for training, evaluating and investigating the model.

.. automodule:: crbm

.. autoclass:: CRBM
    :members:

Sequence-related utilies
===========================

Functions for loading DNA sequences in fasta format
and for convert them to the required *one-hot* encoding.

.. autosummary::

    readSeqsFromFasta
    seqToOneHot
    splitTrainingTest
    
.. automodule:: crbm.sequences
    :members: readSeqsFromFasta, seqToOneHot, splitTrainingTest

Utils
=======

Functions for investigating the results of the :class:`cRBM`,
including for generating position frequency matrices, sequence logos
and clustering plots.

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
            tsneScatterWithPies, violinPlotMotifMatches

Sample dataset
===============

The package contains a small sample dataset consisting
of *Oct4* ChIP-seq sequences of embryonic stem cells from
ENCODE [1]_.

.. autofunction:: crbm.sequences.load_sample

References
===========

.. [1] ENCODE Project Consortium and others. (2012).
        An integrated encyclopedia of DNA elements in the human genome. Nature.
