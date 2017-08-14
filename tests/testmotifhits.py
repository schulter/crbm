from crbm import CRBM, load_sample
import numpy as np
import theano
from theano import tensor as T

def sigmoid(act):
        return 1./(1.+np.exp(-act))

def test_motifhit_dims():
    """Implement tests concerning bottomup activities."""
    # doublestranded = False
    data = load_sample()

    # make theano function
    model = CRBM(num_motifs = 1, motif_length =2)
    encoder=cae.encode()

    encode_fct = theano.function([cae.input], encoder)
    w=lasagne.layers.get_all_layers(model)[1].W.get_value()

    ocuda=encode_fct(seq)

    o = np.zeros(ocuda.shape)
    for s in range(seq.shape[3]-w.shape[3]+1):
        for m in range(w.shape[0]):
            o[0,m,0,s] += \
                    np.sum(np.multiply(w[m,:,0,:], \
                    seq[0, :, 0, s:(s+w.shape[3])]))

    o=sigmoid(o)

    np.testing.assert_allclose(o, ocuda, rtol=1e-5, atol=1e-5)
    # singlestranded
    # doublestranded
    raise Exception("Not yet implemented")

