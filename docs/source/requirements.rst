=============================
System setup
=============================

Requirements
============

The ``secomo`` package is compatible and was tested
with py2.7, py3.4, py3.5 and py3.5.
Prerequisites:

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
results using our set of provided functions (e.g. :func:`secomo.utils.scatterTSNE`).

Finally, if possible, we recommend utilizing 
`CUDA <https://developer.nvidia.com/cuda-downloads>`_. 
`Theano` take advantage of `cuda`, which significantly speeds up the training phase.
See `Theano <http://deeplearning.net/software/theano/>`_ documention for more information.


Installation
============

The secomo package can be obtained from `GitHub <https://github.com/schulter/crbm>`_

At the moment you can install it according to::

    git clone https://github.com/schulter/crbm.git
    cd crbm/
    python setup.py install
