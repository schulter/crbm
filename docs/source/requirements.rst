=============================
System setup
=============================

Requirements
============

To run the ``crbm`` python package the following prerequisites
are required on your system:

* ``theano``
* ``numpy``
* ``scipy``
* ``joblib``
* ``Biopython``
* ``matplotlib``
* ``pandas``
* ``weblogolib``
* ``scikit-learn``
* ``seaborn``


The latter collection of packages are necessary to investigate the model
results using our set of provided functions (e.g. :func:`crbm.utils.scatterTSNE`).

Finally, if possible, we recommend utilizing 
`CUDA <https://developer.nvidia.com/cuda-downloads>`_. 
`Theano` take advantage of `cuda`, which significantly speeds up the training phase.
See `Theano <http://deeplearning.net/software/theano/>`_ documention for more information.


Installation
============

The crbm package can be obtained from `GitHub <https://github.molgen.mpg.de/wkopp/crbm>`_

At the moment you can install it according to::

    git clone https://github.molgen.mpg.de/wkopp/crbm.git
    cd crbm/
    python setup.py install
