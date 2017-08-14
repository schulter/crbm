from crbm import CRBM, load_sample
import numpy as np
import theano
from theano import tensor as T

def sigmoid(act):
        return 1./(1.+np.exp(-act))

def tst_bottomup_activities():
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


def tst_bottomup_probabilities():
    """Tests concerning bottomup probs."""
    raise Exception("Not yet implemented")

def tst_topdown_activities():
    """Tests concerning topdown activities."""
    raise Exception("Not yet implemented")

def tst_topdown_probabilities():
    """Tests concerning topdown activities."""
    raise Exception("Not yet implemented")


def tst_motifhitprobs():
    """Prob dimension."""
    nmotifs = 1
    mlen = 1
    data = load_sample()

    for ds in (True, False):
        for nmot, ml in [(1,1), (2,1), (1,2), (1, data.shape[3])]:
            model = CRBM(nmotifs, mlen, doublestranded = ds)

            #doublestranded = False
            #doublestranded = True
            #Motiflength =0
            #Motiflength =1
            #motiflength=2
            #Motiflength= len seq
            #Motiflength=len seq +1
            p = model.motifHitProbs(data)
            
            d = data.shape
            np.testing.assert_equal(p.shape, (d[0], d[1], 1, d[3]-mlen +1))

def symmetric_motif():
    """All nucleotides with equal prob"""
    seq=oct4[:1,:,:,:]

    # make theano function
    model = cae.model
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

def convolution_correctness_activity():
    """Convolution comparison with alternative approach."""
    
    # doublestranded = False
    seq=oct4[:1,:,:,:]

    # make theano function
    model = cae.model
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


    # doublestranded = True

def bottomup_activities_multiple_motifs():
    seqreader=utils.SeqReader()
    oct4=seqreader.readSequencesFromFile("data/test.fa")
    cae=cAE((4, 10, 15), seqlen = oct4.shape[3], l1_penalty=0,l2_penalty=0)
    seq=oct4[:1,:,:,:]

    def sigmoid(act):
            return 1./(1.+np.exp(-act))

    # make theano function
    model = cae.model
    code=cae.encode()

    encode_fct = theano.function([cae.input], code)
    w=lasagne.layers.get_all_layers(model)[1].W.get_value()

    # run cuda example
    ocuda=encode_fct(seq)

    # run naive example
    o = np.zeros(ocuda.shape)
    for s in range(seq.shape[3]-w.shape[3]+1):
        for m in range(w.shape[0]):
            o[0,m,0,s] += np.sum(np.multiply(w[m,:,0,:], \
                    seq[0, :, 0, s:(s+w.shape[3])]))

    o=sigmoid(o)

    np.testing.assert_allclose(o, ocuda, rtol=1e-5, atol=1e-5)

