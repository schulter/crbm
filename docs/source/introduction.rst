====================
Introduction to cRBM
====================

This package provides functionality to automatically
extract DNA sequence features from a provided set of sequences.
To that end, the cRBM learns redundant DNA sequnce features
in terms of a set of weight matrices.

For a downstream analysis of the features that have been learned
a number of utilities are provided:

1. Convert the cRBM features to **Position frequency matrices**,
   which are commonly used
   to represent transcription factor binding affinities [1]_.
2. Visualize the DNA features in terms of **Sequence logos** [1]_.
3. Visualize the **positional enrichment** of the features on a set of DNA sequences.
4. Visualize the **relative enrichment** of the features 
   across a number of different datasets (e.g. sequences of
   different ChIP-seq experiments; treatment-control).
5. Visualize sequence-based **clustering**.

The tutorial illustrates the main functionality of the package on a
toy example of *Oct4* ChIP-seq peak regions obtained from embryonic stem cells.

Finally, if this tool helps for your analysis, please cite the package::

    @Manual{,
        title = {cRBM: A python package for automatically 
                extracting DNA sequence features},
        author = {Roman Schulte-Sasse, Wolfgang Kopp},
        year = {2017},
    }



References
----------
.. [1] Stormo, Gary D. (2000). 
    DNA binding sites: representation and discovery.
    Bioinformatics.
