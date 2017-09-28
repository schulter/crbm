# Theano imports
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d as conv
import warnings

from theano.sandbox.rng_mrg import MRG_RandomStreams as RS
import scipy

# numpy and python classics
import numpy as np
import time
import joblib
import pprint

"""
This is the actual implementation of our convolutional RBM.
The class implements only contrastive divergence learning so far
but the number of runs for Gibbs Sampling can be varied.
Furthermore, the beforementioned implementation of probabilistic max pooling
computes probabilities for and samples of the hidden layer distribution.
"""


class CRBM:
    """CRBM class.

    The class :class:`CRBM` implements functionality for
    a *convolutional restricted Boltzmann machine* (cRBM) that
    extracts redundant DNA sequence features from a provided set
    of sequences.
    The model can subsequently be used to study the sequence content
    of (e.g. regulatory) sequences, by visualizing the features in terms
    of sequence logos or in order to cluster the sequences based
    on sequence content.

    Parameters
    -----------
    num_motifs : int
        Number of motifs.
    motif_length : int
        Motif length.

    epochs : int
        Number of epochs to train (Default: 100).
    input_dims :int
        Input dimensions aka alphabet size (Default: 4 for DNA).
    doublestranded : bool
        Single strand or both strands. If set to True,
        both strands are scanned. (Default: True).
    batchsize : int
        Batch size (Default: 20).
    learning_rate : float)
        Learning rate (Default: 0.1).
    momentum : float
        Momentum term (Default: 0.95).
    pooling : int
        Pooling factor (not relevant for 
        cRBM, but for future work) (Default: 1).
    cd_k : int
        Number of Gibbs sampling iterations in 
        each persistent contrastive divergence step (Default: 5).
    rho : float
        Target frequency of motif occurrences (Default: 0.01).
    lambda_rate : float
        Sparsity enforcement aka penality term (Default: 0.1).
    """

    def __init__(self, num_motifs, motif_length, epochs = 100, input_dims=4, \
            doublestranded = True, batchsize = 20, learning_rate = 0.1, \
            momentum = 0.95, pooling = 1, cd_k = 5, 
            rho = 0.01, lambda_rate = 0.1):
     
        # sanity checks:
        if num_motifs <= 0:
            raise Exception("Number of motifs must be positive.")

        if motif_length <= 0:
            raise Exception("Motif length must be positive.")

        if epochs < 0:
            raise Exception("Epochs must be non-negative.")

        if input_dims <= 0:
            raise Exception("input_dims must be positive.")
        elif input_dims != 4:
            warnings.warn("input_dims != 4 was not comprehensively \
                tested yet. Be careful when interpreting the results.",
                UserWarning)
        
        if batchsize <= 0:
            raise Exception("batchsize must be positive.")
        
        if learning_rate <= 0.0:
            raise Exception("learning_rate must be positive.")

        if not (momentum >= 0.0 and momentum < 1.):
            raise Exception("momentum must be between zero and one.")

        if pooling <= 0:
            raise Exception("pooling must be positive.")

        if cd_k <= 0:
            raise Exception("cd_k must be positive.")

        if not (rho >= 0.0 and rho < 1.):
            raise Exception("rho must be between zero and one.")

        if lambda_rate < 0.:
            raise Exception("lambda_rate must be non-negative.")

        # parameters for the motifs
        self.num_motifs = num_motifs
        self.motif_length = motif_length
        self.input_dims = input_dims
        self.doublestranded = doublestranded
        self.batchsize = batchsize
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rho = rho
        self.lambda_rate = lambda_rate
        self.pooling = pooling
        self.cd_k = cd_k
        self.epochs = epochs
        self.spmethod = 'entropy'
        self._gradientSparsityConstraint = \
            self._gradientSparsityConstraintEntropy

        x = np.random.randn(self.num_motifs,
                                1,
                                self.input_dims,
                                self.motif_length
                                ).astype(theano.config.floatX)

        self.motifs = theano.shared(value=x, name='W', borrow=True)
        
        # determine the parameter rho for the model if not given
        if not rho:
            rho = 1. / (self.num_motifs * self.motif_length)
            if self.doublestranded:
              rho=rho/2.
            self.rho = rho
        
        # cRBM parameters (2*x to respect both strands of the DNA)
        b = np.zeros((1, self.num_motifs)).astype(theano.config.floatX)

        # adapt the bias such that it will initially have rho motif hits in H
        # That is, we want to have rho percent of the samples positive
        # randn draws from 'standard normal', this is why we have 0 and 1
        b = b + scipy.stats.norm.ppf(self.rho, 0, np.sqrt(self.motif_length))
        self.bias = theano.shared(value=b, name='bias', borrow=True)

        c = np.zeros((1, self.input_dims)).astype(theano.config.floatX)
        self.c = theano.shared(value=c, name='c', borrow=True)

        # infrastructural parameters
        self.theano_rng = RS(seed=int(time.time()))
        self.rng_data_permut = theano.tensor.shared_randomstreams.RandomStreams()
        
        self.motif_velocity = theano.shared(value=np.zeros(self.motifs.get_value().shape).astype(theano.config.floatX),
                                            name='velocity_of_W',
                                            borrow=True)
        self.bias_velocity = theano.shared(value=np.zeros(b.shape).astype(theano.config.floatX),
                                           name='velocity_of_bias',
                                           borrow=True)
        self.c_velocity = theano.shared(value=np.zeros(c.shape).astype(theano.config.floatX),
                                        name='velocity_of_c',
                                        borrow=True)
        
        val = np.zeros((self.batchsize, self.num_motifs, 1, 200)).astype(theano.config.floatX)
        self.fantasy_h = theano.shared(value=val, name='fantasy_h', borrow=True)
        if self.doublestranded:
          self.fantasy_h_prime = theano.shared(value=\
                    np.zeros((self.batchsize, self.num_motifs, 1, 200)).astype(theano.config.floatX), \
                    name='fantasy_h_prime', borrow=True)

        self._compileTheanoFunctions()

    def saveModel(self, filename):
        """Save the model parameters and additional hyper-parameters.

        Parameters
        -----------
        filename : str
            Pickle filename where the model parameters are stored.
        """

        numpyParams = (self.motifs.get_value(),
                self.bias.get_value(),
                self.c.get_value())
        
        hyperparams = ( self.num_motifs,
        self.motif_length, 
        self.input_dims,
        self.doublestranded,
        self.batchsize, 
        self.learning_rate, 
        self.momentum, 
        self.rho, 
        self.lambda_rate, 
        self.pooling, 
        self.cd_k, 
        self.epochs, self.spmethod)

        pickleObject = (numpyParams, hyperparams)
        joblib.dump(pickleObject, filename, protocol= 2)

    @classmethod
    def loadModel(cls, filename):
        """Load a model from a given pickle file.

        Parameters
        -----------
        filename : str
            Pickle file containing the model parameters.
        returns : :class:`CRBM` object
            An instance of CRBM with reloaded parameters.
        """

        numpyParams, hyperparams =joblib.load(filename)
        
        (num_motifs, motif_length, input_dims, \
            doublestranded, batchsize, learning_rate, \
            momentum, rho, lambda_rate,
            pooling, cd_k, 
            epochs, spmethod) = hyperparams

        obj = cls(num_motifs, motif_length, epochs, input_dims, \
                doublestranded, batchsize, learning_rate, \
                momentum, pooling, cd_k,
                rho, lambda_rate)

        motifs, bias, c = numpyParams
        obj.motifs.set_value(motifs)
        obj.bias.set_value(bias)
        obj.c.set_value(c)
        return obj

    def _bottomUpActivity(self, data, flip_motif=False):
        """Theano function for computing bottom up activity."""

        out = conv(data, self.motifs, filter_flip=flip_motif)
        out = out + self.bias.dimshuffle('x', 1, 0, 'x')
        return out

    def _bottomUpProbability(self,activities):
        """Theano function for computing bottom up Probability."""

        pool = self.pooling
        x = activities.reshape((activities.shape[0], \
                activities.shape[1], activities.shape[2], \
                activities.shape[3]//pool, pool))
        norm=T.sum(1. +T.exp(x), axis=4,keepdims=True)
        x=T.exp(x)/norm
        x=x.reshape((activities.shape[0], \
                activities.shape[1], activities.shape[2], \
                activities.shape[3]))
        return x
        
    def _bottomUpSample(self,probs):
        """Theano function for bottom up sampling."""

        pool = self.pooling
        _probs=probs.reshape((probs.shape[0], probs.shape[1], probs.shape[2], probs.shape[3]//pool, pool))
        _probs_reshape=_probs.reshape((_probs.shape[0]*_probs.shape[1]*_probs.shape[2]*_probs.shape[3],pool))
        samples=self.theano_rng.multinomial(pvals=_probs_reshape)
        samples=samples.reshape((probs.shape[0],probs.shape[1],probs.shape[2],probs.shape[3]))
        return T.cast(samples,theano.config.floatX)

    def _computeHgivenV(self, data, flip_motif=False):
        """Theano function for complete bottom up pass."""

        activity=self._bottomUpActivity(data,flip_motif)
        probability=self._bottomUpProbability(activity)
        sample=self._bottomUpSample(probability)
        return [probability, sample]

    def _topDownActivity(self, h, hprime):
        """Theano function for top down activity."""
        W = self.motifs.dimshuffle(1, 0, 2, 3)
        C = conv(h, W, border_mode='full', filter_flip=True)
        
        out = T.sum(C, axis=1, keepdims=True)  # sum over all K

        if hprime:
            C = conv(hprime, W[:,:,::-1,::-1], \
                    border_mode='full', filter_flip=True)
            out = out+ T.sum(C, axis=1, keepdims=True)  # sum over all K

        c_bc = self.c
        c_bc = c_bc.dimshuffle('x', 0, 1, 'x')
        activity = out + c_bc
        return activity

    def _topDownProbability(self, activity, softmaxdown = True):
        """Theano function for top down probability."""
        if softmaxdown:
            return self._softmax(activity)
        else:
            return 1./(1.-T.exp(-activity))

    def _topDownSample(self, probability, softmaxdown = True):
        """Theano function for top down sample."""
        if softmaxdown:
            pV_ = probability.dimshuffle(0, 1, 3, 2).reshape( \
                (probability.shape[0]*probability.shape[3], 
                    probability.shape[2]))
            V_ = self.theano_rng.multinomial(n=1, pvals=pV_).astype(
                    theano.config.floatX)
            V = V_.reshape((probability.shape[0], 1, probability.shape[3], 
                probability.shape[2])).dimshuffle(0, 1, 3, 2)

        else:
            V=self.theano_rng.multinomial(n=1,\
                pvals=probability).astype(theano.config.floatX)
        return V

    def _computeVgivenH(self, H_sample, H_sample_prime, softmaxdown=True):
        """Theano function for complete top down pass."""

        activity = self._topDownActivity(H_sample, H_sample_prime)

        prob = self._topDownProbability(activity, softmaxdown)
        sample = self._topDownSample(prob, softmaxdown)

        return [prob, sample]

    def _collectVHStatistics(self, prob_of_H, data):
        """Theano function for collecting V*H statistics."""

        # reshape input
        data = data.dimshuffle(1, 0, 2, 3)
        prob_of_H = prob_of_H.dimshuffle(1, 0, 2, 3)
        avh = conv(data, prob_of_H, border_mode="valid", filter_flip=False)
        avh = avh / T.prod(prob_of_H.shape[1:])
        avh = avh.dimshuffle(1, 0, 2, 3).astype(theano.config.floatX)

        return avh

    def _collectVStatistics(self, data):
        """Theano function for collecting V statistics."""

        # reshape input
        a = T.mean(data, axis=(0, 1, 3)).astype(theano.config.floatX)
        a = a.dimshuffle('x', 0)
        a = T.inc_subtensor(a[:, :], a[:, ::-1])  # match a-t and c-g occurances

        return a

    def _collectHStatistics(self, data):
        """Theano function for collecting H statistics."""

        # reshape input
        a = T.mean(data, axis=(0, 2, 3)).astype(theano.config.floatX)
        a = a.dimshuffle('x', 0)

        return a

    def _collectUpdateStatistics(self, prob_of_H, prob_of_H_prime, data):
        """Theano function for collecting the complete update statistics."""

        average_VH = self._collectVHStatistics(prob_of_H, data)
        average_H = self._collectHStatistics(prob_of_H)

        if prob_of_H_prime:
            average_VH_prime=self._collectVHStatistics(prob_of_H_prime, data)
            average_H_prime=self._collectHStatistics(prob_of_H_prime)
            average_VH=(average_VH+average_VH_prime[:,:,::-1,::-1])/2.
            average_H=(average_H+average_H_prime)/2.

        average_V = self._collectVStatistics(data)
        return average_VH, average_H, average_V
    
    def _updateWeightsOnMinibatch(self, D, gibbs_chain_length):
        """Theano function that defines an SGD update step with momentum."""

        # calculate the data gradient for weights (motifs), bias and c
        [prob_of_H_given_data,H_given_data] = self._computeHgivenV(D)

        if self.doublestranded:
            [prob_of_H_given_data_prime,H_given_data_prime] = \
                    self._computeHgivenV(D, True)
        else:
            [prob_of_H_given_data_prime,H_given_data_prime] = [None, None]

        # calculate data gradients
        G_motif_data, G_bias_data, G_c_data = \
                  self._collectUpdateStatistics(prob_of_H_given_data, \
                  prob_of_H_given_data_prime, D)

        # calculate model probs
        H_given_model = self.fantasy_h
        if self.doublestranded:
            H_given_model_prime = self.fantasy_h_prime
        else:
            H_given_model_prime = None

        for i in range(gibbs_chain_length):
            prob_of_V_given_model, V_given_model = \
                    self._computeVgivenH(H_given_model, H_given_model_prime)
            #sample up
            prob_of_H_given_model, H_given_model = \
                    self._computeHgivenV(V_given_model)

            if self.doublestranded:
                prob_of_H_given_model_prime, H_given_model_prime = \
                        self._computeHgivenV(V_given_model,  True)
            else:
                prob_of_H_given_model_prime, H_given_model_prime = None, None
        
        # compute the model gradients
        G_motif_model, G_bias_model, G_c_model = \
                  self._collectUpdateStatistics(prob_of_H_given_model, \
                  prob_of_H_given_model_prime, V_given_model)
        
        mu = self.momentum
        alpha = self.learning_rate
        sp = self.lambda_rate
        reg_motif, reg_bias = self._gradientSparsityConstraint(D)

        vmotifs = mu * self.motif_velocity + \
                alpha * (G_motif_data - G_motif_model-sp*reg_motif)
        vbias = mu * self.bias_velocity + \
                alpha * (G_bias_data - G_bias_model-sp*reg_bias)
        vc = mu*self.c_velocity + \
                alpha * (G_c_data - G_c_model)

        new_motifs = self.motifs + vmotifs
        new_bias = self.bias + vbias
        new_c = self.c + vc

        
        updates = [(self.motifs, new_motifs), (self.bias, new_bias), (self.c, new_c),
                   (self.motif_velocity, vmotifs), (self.bias_velocity, vbias), (self.c_velocity, vc),
                   (self.fantasy_h, H_given_model)]
        if self.doublestranded:
            updates.append((self.fantasy_h_prime, H_given_model_prime))

        return updates

    def _gradientSparsityConstraintEntropy(self, data):
        """Theano function that defines the entropy-based sparsity constraint."""
        # get expected[H|V]
        [prob_of_H, _] = self._computeHgivenV(data)
        q = self.rho
        p = T.mean(prob_of_H, axis=(0, 2, 3))

        gradKernels = - T.grad(T.mean(q*T.log(p) + (1-q)*T.log(1-p)),
                             self.motifs)
        gradBias = - T.grad(T.mean(q*T.log(p) + (1-q)*T.log(1-p)),
                          self.bias)
        return gradKernels, gradBias

    def _compileTheanoFunctions(self):
        """This methods compiles all theano functions."""

        print("Start compiling Theano training function...")
        D = T.tensor4('data')
        updates = self._updateWeightsOnMinibatch(D, self.cd_k)
        self.theano_trainingFct = theano.function(
              [D],
              None,
              updates=updates,
              name='train_CRBM'
        )

        #compute mean free energy
        mfe_ = self._meanFreeEnergy(D)
        #compute number  of motif hits
        [_, H] = self._computeHgivenV(D)
        
        #H = self.bottomUpProbability(self.bottomUpActivity(D))
        nmh_=T.mean(H)  # mean over samples (K x 1 x N_h)


        #compute norm of the motif parameters
        twn_=T.sqrt(T.mean(self.motifs**2))
        
        #compute information content
        pwm = self._softmax(self.motifs)
        entropy = -pwm * T.log2(pwm)
        entropy = T.sum(entropy, axis=2)  # sum over letters
        ic_= T.log2(self.motifs.shape[2]) - \
            T.mean(entropy)  # log is possible information due to length of sequence
        medic_= T.log2(self.motifs.shape[2]) - \
            T.mean(T.sort(entropy, axis=2)[:, :, entropy.shape[2] // 2])
        self.theano_evaluateData = theano.function(
              [D],
              [mfe_, nmh_],
              name='evaluationData'
        )

        W=T.tensor4("W")
        self.theano_evaluateParams = theano.function(
              [],
              [twn_,ic_,medic_],
                givens={W:self.motifs},
              name='evaluationParams'
        )
        fed=self._freeEnergyForData(D)
        self.theano_freeEnergy=theano.function( [D],fed,name='fe_per_datapoint')

        fed=self._freeEnergyPerMotif(D)
        self.theano_fePerMotif=theano.function( [D],fed,name='fe_per_motif')

        
        if self.doublestranded:
            self.theano_getHitProbs = theano.function([D], \
                self._bottomUpProbability(self._bottomUpActivity(D)))
        else:
            self.theano_getHitProbs = theano.function([D], \
                #self.bottomUpProbability( T.maximum(self.bottomUpActivity(D),
                self._bottomUpProbability( self._bottomUpActivity(D) + 
                        self._bottomUpActivity(D, True)))
        print("Compilation of Theano training function finished")

    def _evaluateData(self, data):
        """Evaluate performance on given numpy array.
        
        This is used to monitor training progress.
        """
        return self.theano_evaluateData(data)

    def _trainingFct(self, data):
        """Train on mini-batch given numpy array."""
        return self.theano_trainingFct(data)

    def _evaluateParams(self):
        """Evaluate parameters.

        This is used to monitor training progress.
        """
        return self.theano_evaluateParams()

    def motifHitProbs(self, data):
        """Motif match probabilities.

        Parameters
        -----------
        data : numpy-array
            4D numpy array representing a DNA sequence in one-hot encoding.
            See :meth:`crbm.sequences.seqToOneHot`.

        returns : numpy-array
            Per-position motif match probabilities of all motifs as numpy array.
        """
        return self.theano_getHitProbs(data)

    def freeEnergy(self, data):
        """Free energy determined on the given dataset.

        Parameters
        -----------
        data : numpy-array
            4D numpy array representing a DNA sequence in one-hot encoding.
            See :meth:`crbm.sequences.seqToOneHot`.

        returns : numpy-array
            Free energy per sequence.
        """
        return self.theano_freeEnergy(data)

    def fit(self, training_data, test_data = None):
        """Fits the cRBM to the provided training sequences.

        Parameters
        -----------
        training_data : numpy-array
            4D-Numpy array representing the training sequence in one-hot encoding.
            See :meth:`crbm.sequences.seqToOneHot`.

        test_data : numpy-array
            4D-Numpy array representing the validation sequence in one-hot encoding.
            If no test_data is provided, the training progress will be reported
            on the training set itself. See :meth:`crbm.sequences.seqToOneHot`.
        """
        # assert that pooling can be done without rest to the division
        # compute sequence length
        nseq=int((training_data.shape[3]-\
            self.motif_length + 1)/\
            self.pooling)*\
            self.pooling+ \
            self.motif_length -1
        training_data=training_data[:,:,:,:nseq]

        if type(test_data) != type(None):
            nseq=int((test_data.shape[3]-\
                self.motif_length + 1)/\
                self.pooling)*\
                self.pooling+ \
                self.motif_length -1
            test_data=test_data[:,:,:,:nseq]
        else:
            test_data = training_data

        # some debug printing
        numTrainingBatches = training_data.shape[0] / self.batchsize
        numTestBatches = test_data.shape[0] / self.batchsize
        print(("BatchSize: " + str(self.batchsize)))
        print(("Num of iterations per epoch: " + str(numTrainingBatches)))
        start = time.time()

        # compile training function

        # now perform training
        print("Start training the model...")
        starttime = time.time()

        for epoch in range(self.epochs):
            for [start,end] in self._iterateBatchIndices(\
                            training_data.shape[0],self.batchsize):
                self._trainingFct(training_data[start:end,:,:,:])
            meanfe=0.0
            meannmh=0.0
            nb=0
            for [start,end] in self._iterateBatchIndices(\
                            test_data.shape[0],self.batchsize):
                [mfe_,nmh_]=self._evaluateData(test_data[start:end,:,:,:])
                meanfe=meanfe+mfe_
                meannmh=meannmh+nmh_
                nb=nb+1
            [twn_,ic_,medic_]=self._evaluateParams()
            #for batchIdx in range(numTestBatches):
            print(("Epoch {:d}: ".format(epoch) + \
                    "FE={:1.3f} ".format(meanfe/nb) + \
                    "NumH={:1.4f} ".format(meannmh/nb) + \
                    "WNorm={:2.2f} ".format(float(twn_)) + \
                    "IC={:1.3f} medIC={:1.3f}".format(float(ic_), float(medic_))))

        # done with training
        print(("Training finished after: {:5.2f} seconds!".format(\
                time.time()-starttime)))

    def _meanFreeEnergy(self, D):
        """Theano function for computing the mean free energy."""
        return T.sum(self._freeEnergyForData(D))/D.shape[0]
        
    def getPFMs(self):
        """Returns the weight matrices converted to *position frequency matrices*.

        Parameters
        -----------
        returns: numpy-array
            List of position frequency matrices as numpy arrays.
        """

        def softmax_(x):
            x_exp = np.exp(x)
            y = np.zeros(x.shape)
            for i in range(x.shape[1]):
                y[:,i] = x_exp[:,i] / np.sum(x_exp[:,i])
            return y
        return [ softmax_(m[0, :, :]) for m in self.motifs.get_value() ]

    def _freeEnergyForData(self, D):
        """Theano function for computing the free energy (per position)."""

        pool = self.pooling

        x=self._bottomUpActivity(D)

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3]//pool, pool))
        free_energy = -T.sum(T.log(1.+T.sum(T.exp(x), axis=4)), axis=(1, 2, 3))
        if self.doublestranded:
            x=self._bottomUpActivity(D,True)
  
            x = x.reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3]//pool, pool))
            free_energy = free_energy -T.sum(T.log(1.+T.sum(T.exp(x), axis=4)), axis=(1, 2, 3))
        
        cMod = self.c
        cMod = cMod.dimshuffle('x', 0, 1, 'x')  # make it 4D and broadcastable there
        free_energy = free_energy - T.sum(D * cMod, axis=(1, 2, 3))
        
        return free_energy/D.shape[3]

    def _freeEnergyPerMotif(self, D):
        """Theano function for computing the free energy (per motif)."""

        pool = self.pooling

        x=self._bottomUpActivity(D)

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3]//pool, pool))
        free_energy = -T.sum(T.log(1.+T.sum(T.exp(x), axis=4)), axis=(2, 3))

        if self.doublestranded:
            x=self._bottomUpActivity(D,True)
            x = x.reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3]//pool, pool))
            free_energy = free_energy -T.sum(T.log(1.+T.sum(T.exp(x), axis=4)), axis=(2, 3))
        
        cMod = self.c
        cMod = cMod.dimshuffle('x', 0, 1, 'x')  # make it 4D and broadcastable there
        free_energy = free_energy - T.sum(D * cMod, axis=(1, 2, 3)).dimshuffle(0, 'x')
        
        return free_energy

    def _softmax(self, x):
        """Softmax operation."""

        return T.exp(x) / T.exp(x).sum(axis=2, keepdims=True)

    def __repr__(self):
        st = "Parameters:\n\n"
        st += "Number of motifs: {}\n".format(self.num_motifs)
        st += "Motif length: {}\n".format(self.motif_length)
        st += "\n"
        st += "Hyper-parameters:\n\n"
        st += "input dims: {:d}".format(self.input_dims)
        st += "doublestranded: {}".format(self.doublestranded)
        st += "batchsize: {:d}".format(self.batchsize)
        st += "learning rate: {:1.3f}".format(self.learning_rate)
        st += "momentum: {:1.3f}".format(self.momentum)
        st += "rho: {:1.4f}".format(self.rho)
        st += "lambda: {:1.3f}".format(self.lambda_rate)
        st += "pooling: {:d}".format(self.pooling)
        st += "cd_k: {:d}".format(self.cd_k)
        st += "epochs: {:d}".format(self.epochs)
        return st

    def _iterateBatchIndices(self, totalsize,nbatchsize):
        """Returns indices in batches."""

        return [ [i,i+nbatchsize] if i+nbatchsize<=totalsize \
                    else [i,totalsize] for i in range(totalsize)[0::nbatchsize] ]
    
