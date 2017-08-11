====================
Introduction to cRBM
====================

This package provides functionality to automatically
extract DNA sequence features from a provided set of sequences.
To that end, the cRBM learns redundant DNA sequnce features
in terms of a set of weight matrices.
Those weight matrices can be

1. converted to **position frequency matrices**, which are commonly used
   to represent transcription factor binding affinities [1]_,
   for subsequent analysis steps.
2. the DNA features can be visualized in terms of sequence logos [1]_.
3. the relative enrichment of the features can be investigated across
   a number of different samples (e.g. different ChIP-seq experiments).
4. the sequence can be clustered based on their sequence composition.

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
