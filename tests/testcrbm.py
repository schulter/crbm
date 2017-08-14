from crbm import CRBM, load_sample
import pytest
import numpy as np
import theano
from theano import tensor as T

class TestCRBM(object):
    data = load_sample()

    def test_crbm_parametervalues(self):

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

    def test_crbm_creation(self):

        # cRBM
        model = CRBM(num_motifs = 2, motif_length = 5)

        W = model.motifs.get_value()
        b = model.bias.get_value()
        c = model.c.get_value()

        np.testing.assert_equal(W.shape, (2, 1, 4, 5))
        np.testing.assert_equal(b.shape, (1,2))
        np.testing.assert_equal(c.shape, (1,4))

        # save model
        model.saveModel("model.pkl")

        model2 = CRBM.loadModel("model.pkl")

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
        # doublestranded = False

        data = self.data[:100]
        mnum = 10
        mlen = 15

        for ds in (True, False):
            model = CRBM(num_motifs = mnum, motif_length = mlen, 
                    doublestranded = ds, epochs = 1)

            pred = model.motifHitProbs(data)

            np.testing.assert_equal((data.shape[0], mnum, 1, data.shape[3]-mlen+1), pred.shape)
