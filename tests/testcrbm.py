from secomo import CRBM, load_sample
import pytest
import numpy as np
import theano
from theano import tensor as T
import os

def sigmoid(act):
        return 1./(1.+np.exp(-act))

class TestCRBM(object):
    data = load_sample()

    def test_crbm_parametervalues(self):
        """Test CRBM init parameters."""

        with pytest.raises(Exception):
            CRBM(num_motifs = 0, motif_length = 5)

        with pytest.raises(Exception):
            CRBM(num_motifs = 1, motif_length = 0)

        with pytest.raises(Exception):
            CRBM(num_motifs = 1, motif_length = 1, epochs = -1)


        with pytest.raises(Exception):
            CRBM(num_motifs = 1, motif_length = 1, input_dims = -1)

        with pytest.warns(UserWarning):
            CRBM(num_motifs = 1, motif_length = 1, input_dims = 3)

        with pytest.raises(Exception):
            CRBM(num_motifs = 1, motif_length = 1, batchsize = 0)

        with pytest.raises(Exception):
            CRBM(num_motifs = 1, motif_length = 1, learning_rate = 0.0)

        with pytest.raises(Exception):
            CRBM(num_motifs = 1, motif_length = 1, momentum = -1.)
        with pytest.raises(Exception):
            CRBM(num_motifs = 1, motif_length = 1, momentum = 1.1)

        with pytest.raises(Exception):
            CRBM(num_motifs = 1, motif_length = 1, pooling = 0)

        with pytest.raises(Exception):
            CRBM(num_motifs = 1, motif_length = 1, cd_k = 0)

        with pytest.raises(Exception):
            CRBM(num_motifs = 1, motif_length = 1, rho = -1.)
        with pytest.raises(Exception):
            CRBM(num_motifs = 1, motif_length = 1, rho = 1.1)

        with pytest.raises(Exception):
            CRBM(num_motifs = 1, motif_length = 1, lambda_rate = -.1)

    def test_crbm_creation(self, tmpdir):
        """Test CRBM creation and reloading."""

        # cRBM
        model = CRBM(num_motifs = 2, motif_length = 5)

        W = model.motifs.get_value()
        b = model.bias.get_value()
        c = model.c.get_value()

        np.testing.assert_equal(W.shape, (2, 1, 4, 5))
        np.testing.assert_equal(b.shape, (1,2))
        np.testing.assert_equal(c.shape, (1,4))

        # save model
        model.saveModel(os.path.join(tmpdir.strpath, "model.pkl"))

        model2 = CRBM.loadModel(os.path.join(tmpdir.strpath,"model.pkl"))

        # check hyper-parameters

        np.testing.assert_equal(model.num_motifs, model2.num_motifs)
        np.testing.assert_equal(model.motif_length, model2.motif_length)
        np.testing.assert_equal(model.epochs, model2.epochs)
        np.testing.assert_equal(model.input_dims, model2.input_dims)
        np.testing.assert_equal(model.doublestranded, model2.doublestranded)
        np.testing.assert_equal(model.batchsize, model2.batchsize)
        np.testing.assert_equal(model.momentum, model2.momentum)
        np.testing.assert_equal(model.pooling, model2.pooling)
        np.testing.assert_equal(model.cd_k, model2.cd_k)
        np.testing.assert_equal(model.rho, model2.rho)
        np.testing.assert_equal(model.lambda_rate, model2.lambda_rate)

        # check model parameters

        np.testing.assert_allclose(model.motifs.get_value(),
                                    model2.motifs.get_value())
        np.testing.assert_allclose(model.bias.get_value(),
                                    model2.bias.get_value())
        np.testing.assert_allclose(model.c.get_value(),
                                    model2.c.get_value())

    def test_training(self):
        """Test training.

        There is nothing to validate numerically,
        just run to catch syntax errors
        """
        # doublestranded = False

        data = self.data[:100]

        for ds in (True, False):
            model = CRBM(num_motifs = 10, motif_length = 15,
                    doublestranded = ds, epochs = 1)
            model.fit(data)
            model.fit(data, data[:10])

    def test_motifhitprobs_dims(self):
        """Motif hit prob dims."""

        data = self.data[:100]
        mnum = 10
        mlen = 15

        for ds in (True, False):
            model = CRBM(num_motifs = mnum, motif_length = mlen,
                    doublestranded = ds, epochs = 1)

            pred = model.motifHitProbs(data)

            np.testing.assert_equal((data.shape[0], mnum, 1, data.shape[3]-mlen+1), pred.shape)

    def test_pfms(self):
        """Test PFM dimensions and correctness."""

        mnum = 10
        mlen = 15

        for ds in (True, False):
            model = CRBM(num_motifs = mnum, motif_length = mlen,
                    doublestranded = ds, epochs = 1)

            pfms = model.getPFMs()

            np.testing.assert_equal(len(pfms), mnum)

            for i in range(mnum):
                np.testing.assert_allclose(pfms[i].sum(), mlen)

    def test_bottomup_singlestranded(self):
        self.bottomup(False)

    def test_bottomup_doublestranded(self):
        self.bottomup(True)

    def bottomup(self, flip):
        """Tests bottomup activities on toy example."""

        data = self.data[:11]
        nmot = 10
        mlen = 5

        # make theano function
        model = CRBM(num_motifs = nmot, motif_length = mlen)
        input = T.tensor4()

        activ = theano.function([input],
                model._bottomUpActivity(input, flip))
        prob = theano.function([input],
                model._bottomUpProbability(
                    model._bottomUpActivity(input, flip)))
        hgivenv = theano.function([input],
                model._computeHgivenV(input, flip))

        if flip:
            w = model.motifs.get_value()[:,:,::-1,::-1]
        else:
            w = model.motifs.get_value()

        b = model.bias.get_value()

        output = activ(data)

        output_control = np.zeros(output.shape)
        for seq in range(data.shape[0]):
            for s in range(data.shape[3]-w.shape[3]+1):
                for m in range(w.shape[0]):
                    output_control[seq,m,0,s] += \
                            np.multiply(w[m,0,:,:], \
                            data[seq, 0, :, s:(s+w.shape[3])]).sum() \
                            + b[0,m]

        np.testing.assert_allclose(output, output_control, rtol=1e-5, atol=1e-5)

        np.testing.assert_allclose(sigmoid(output_control), prob(data),
                rtol=1e-5, atol=1e-5)

        output, _ = hgivenv(data)
        np.testing.assert_allclose(output, sigmoid(output_control),
                rtol=1e-5, atol=1e-5)
        np.testing.assert_equal(output.shape, (data.shape[0], nmot,
            1, data.shape[3] - mlen +1))

    def controlTopDownActivity(self, w, c, data, datap = None):
        """Top down activity control implementation."""

        seqlen = data.shape[3]+w.shape[3]-1
        nseq = data.shape[0]
        nmot = w.shape[0]
        mlen = w.shape[3]

        output_control = np.zeros((nseq, 1, 4, seqlen))
        output_control += c[np.newaxis, 0,:, np.newaxis]
        for seq in range(nseq):
            for pos in range(seqlen-mlen, -1, -1):
                for m in range(nmot):
                    output_control[seq,0,:,pos:(pos+mlen)] += \
                            w[m,0,:,:]*data[seq, m, 0, pos] + \
                            w[m,0,::-1,::-1]*( \
                                0.0 if type(datap)==type(None) \
                                        else datap[seq, m, 0, pos])
        return output_control

    def topdownActivitySS(self, model, data):

        input = T.tensor4()

        vact = theano.function([input], model._topDownActivity(input, None))
        vprob = theano.function([input], model._topDownProbability(
            model._topDownActivity(input, None)))
        vgivenh = theano.function([input], model._computeVgivenH(input, None))

        out_act = vact(data)

        out_prob = vprob(data)

        out_prob2, _ = vgivenh(data)

        return out_act, out_prob, out_prob2

    def topdownActivityDS(self, model, data, datap):

        i1 = T.tensor4()
        i2 = T.tensor4()

        vact = theano.function([i1, i2], model._topDownActivity(i1, i2))
        vprob = theano.function([i1, i2], model._topDownProbability(
            model._topDownActivity(i1, i2)))
        vgivenh = theano.function([i1, i2], model._computeVgivenH(i1, i2))

        out_act = vact(data, datap)

        out_prob = vprob(data, datap)

        out_prob2, _ = vgivenh(data, datap)

        return out_act, out_prob, out_prob2

    def test_topdown_singlestranded_full(self):
        """Test computeVgivenH with DNA as input."""

        data = self.data[:11]
        nmot = 10
        mlen = 5

        # make theano function
        model = CRBM(num_motifs = nmot, motif_length = mlen)
        input = T.tensor4()

        _, h1 = model._computeHgivenV(input, False)

        vgivenh = theano.function([input], model._computeVgivenH(h1, None))

        w = model.motifs.get_value()

        c = model.c.get_value()

        poutput, soutput = vgivenh(data)

        # test correct shape
        np.testing.assert_equal(poutput.shape, data.shape)

        # test correct normalization
        # each position sums to one
        np.testing.assert_allclose(poutput.sum(), data.shape[0]*data.shape[3],
                rtol=1e-4, atol=1e-4)
        # test correct sampling: one 1 per position
        np.testing.assert_allclose(soutput.sum(), data.shape[0]*data.shape[3],
                rtol=1e-4, atol=1e-4)

    def test_topdown_doublestranded_full(self):
        """Test computeVgivenH with DNA as input."""

        data = self.data[:11]
        nmot = 10
        mlen = 5

        # make theano function
        model = CRBM(num_motifs = nmot, motif_length = mlen)
        i1 = T.tensor4()

        _, h1 = model._computeHgivenV(i1, False)
        _, h2 = model._computeHgivenV(i1, True)

        vgivenh = theano.function([i1], model._computeVgivenH(h1, h2))

        w = model.motifs.get_value()

        c = model.c.get_value()

        poutput, soutput = vgivenh(data)

        np.testing.assert_equal(poutput.shape, data.shape)

        np.testing.assert_allclose(poutput.sum(), data.shape[0]*data.shape[3],
                rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(soutput.sum(), data.shape[0]*data.shape[3],
                rtol=1e-4, atol=1e-4)


    def test_topdown_singlestranded_onlyzeros(self):
        """ Test VgivenH with only zero hiddens."""

        nseq = 11
        seqlen = 200
        nmot = 10
        mlen = 5

        data = np.zeros((nseq, nmot, 1, seqlen-mlen+1), dtype="float32")

        # make theano function
        model = CRBM(num_motifs = nmot, motif_length = mlen)

        oa, op, op2 = self.topdownActivitySS(model, data)

        # test correct shape
        np.testing.assert_equal(op.shape, (nseq, 1, 4, seqlen))
        np.testing.assert_equal(op.shape, op2.shape)
        np.testing.assert_equal(op, op2)

        # all must be 0.25
        np.testing.assert_allclose(0.25, op, rtol=1e-5, atol=1e-5)

    def test_topdown_doublestranded_onlyzeros(self):
        """ Test VgivenH with only zero hiddens."""

        nseq = 11
        seqlen = 200
        nmot = 10
        mlen = 5

        data = np.zeros((nseq, nmot, 1, seqlen-mlen+1), dtype="float32")

        # make theano function
        model = CRBM(num_motifs = nmot, motif_length = mlen)

        oa, op, op2 = self.topdownActivityDS(model, data, data)

        # test correct shape
        np.testing.assert_equal(op.shape, (nseq, 1, 4, seqlen))
        np.testing.assert_equal(op.shape, op2.shape)
        np.testing.assert_equal(op, op2)

        # all must be 0.25
        np.testing.assert_allclose(0.25, op, rtol=1e-5, atol=1e-5)

    def test_topdown_singlestranded_onlyones(self):
        """ Test VgivenH with only ones hiddens."""

        nseq = 11
        seqlen = 200
        nmot = 10
        mlen = 5

        data = np.ones((nseq, nmot, 1, seqlen-mlen+1), dtype="float32")

        # make theano function
        model = CRBM(num_motifs = nmot, motif_length = mlen)

        oa, op, op2 = self.topdownActivitySS(model, data)

        w = model.motifs.get_value()
        c = model.c.get_value()
        output_control = self.controlTopDownActivity(w, c, data)

        np.testing.assert_equal(oa.shape, (nseq, 1, 4, seqlen))
        np.testing.assert_equal(op.shape, (nseq, 1, 4, seqlen))
        np.testing.assert_equal(op2.shape, (nseq, 1, 4, seqlen))

        # check if all activities are similar
        np.testing.assert_allclose(oa, output_control,
                rtol=1e-5, atol=1e-5)


        output_control = np.exp(output_control) / np.exp( \
                output_control).sum(axis=2, keepdims=True)

        # check if all probabilities are similar
        np.testing.assert_allclose(output_control, op, rtol=1e-5, atol=1e-5)

        # check correctness of _computeVgivenH
        np.testing.assert_allclose(op, op2, rtol=1e-5, atol=1e-5)

    def test_topdown_doublestranded_onlyones(self):
        """ Test VgivenH with only ones hiddens."""

        nseq = 11
        seqlen = 200
        nmot = 10
        mlen = 5

        data = np.ones((nseq, nmot, 1, seqlen-mlen+1), dtype="float32")

        # make theano function
        model = CRBM(num_motifs = nmot, motif_length = mlen)

        oa, op, op2 = self.topdownActivityDS(model, data, data)

        w = model.motifs.get_value()
        c = model.c.get_value()

        output_control = self.controlTopDownActivity(w, c, data, data)

        np.testing.assert_equal(oa.shape, (nseq, 1, 4, seqlen))

        # check if all activities are similar
        np.testing.assert_allclose(oa, output_control, rtol=1e-5, atol=1e-5)

        output_control = np.exp(output_control) / np.exp( \
                output_control).sum(axis=2, keepdims=True)

        # check if all probabilities are similar
        np.testing.assert_allclose(output_control, op, rtol=1e-5, atol=1e-5)

        np.testing.assert_allclose(op, op2, rtol=1e-5, atol=1e-5)

    def test_topdown_singlestranded_randomhidden(self):
        """Test topdown with random hidden states."""

        nseq = 11
        seqlen = 200
        nmot = 10
        mlen = 5

        data = np.random.binomial(1, p = \
                [[[[0.1]*(seqlen-mlen+1)]*1]*nmot]*nseq).astype("float32")

        # make theano function
        model = CRBM(num_motifs = nmot, motif_length = mlen)

        oa, op, op2 = self.topdownActivitySS(model, data)

        w = model.motifs.get_value()
        c = model.c.get_value()

        output_control = self.controlTopDownActivity(w, c, data)

        np.testing.assert_equal(oa.shape, (nseq, 1, 4, seqlen))
        np.testing.assert_equal(op.shape, (nseq, 1, 4, seqlen))

        np.testing.assert_allclose(oa, output_control, rtol=1e-5, atol=1e-5)

        output_control = np.exp(output_control) / np.exp( \
                output_control).sum(axis=2, keepdims=True)
        np.testing.assert_allclose(output_control, op, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(output_control, op2, rtol=1e-5, atol=1e-5)


    def test_topdown_doublestranded_randomhidden(self):
        """Test topdown with random hidden states."""

        nseq = 11
        seqlen = 200
        nmot = 10
        mlen = 5

        data = np.random.binomial(1, p = \
                [[[[0.1]*(seqlen-mlen+1)]*1]*nmot]*nseq).astype("float32")
        datap = np.random.binomial(1, p = \
                [[[[0.1]*(seqlen-mlen+1)]*1]*nmot]*nseq).astype("float32")

        # make theano function
        model = CRBM(num_motifs = nmot, motif_length = mlen)

        oa, op, op2 = self.topdownActivityDS(model, data, datap)

        w = model.motifs.get_value()
        c = model.c.get_value()

        output_control = self.controlTopDownActivity(w, c, data, datap)


        np.testing.assert_equal(oa.shape, (nseq, 1, 4, seqlen))
        np.testing.assert_equal(op.shape, (nseq, 1, 4, seqlen))
        np.testing.assert_equal(op2.shape, (nseq, 1, 4, seqlen))

        np.testing.assert_allclose(oa, output_control, rtol=1e-5, atol=1e-5)

        print(oa.shape)
        print(output_control.shape)

        output_control = np.exp(output_control) / np.exp( \
                output_control).sum(axis=2, keepdims = True)
        print(output_control.shape)
        np.testing.assert_allclose(output_control, op, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(op, op2, rtol=1e-5, atol=1e-5)


    def test_freeEnergy_dims(self):

        data = self.data[:11]
        nmot = 10
        mlen = 5

        # make theano function
        model = CRBM(num_motifs = nmot, motif_length = mlen)

        fe = model.freeEnergy(data)

        print(fe.shape)
        np.testing.assert_equal(fe.shape, (data.shape[0],))
