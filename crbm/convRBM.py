
# Theano imports
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d as conv

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
    """
    Initialize the cRBM. The parameters here are global params that should not change
    during the execution of training or testing and characterize the network.

    Parameters:
    _motifLength:    How long are the motifs (position weight matrices PWM). This
                     This is equivalent to ask what the number of k-mers is.
                     The current approach only deals with one fixed motif length.
             
    _numMotifs:      How many motifs are applied to the sequence, that is how many
                     hidden units does the network have. Each hidden unit consists
                     of a vector of size (sequenceLength-motifLength+1)
                     
    _poolingFactor:  How many units from the hidden layer are pooled together.
                     Note that the number has to divide evenly to the length of
                     the hidden units, that is:
                     mod(sequenceLength-motifLength+1, poolingFactor) == 0
                     (1 = equivalent to sigmoid activation)
    """
    def __init__(self, num_motifs, motif_length, epochs = 100, input_dims=4, \
            doublestranded = True, batchsize = 20, learning_rate = 0.1, \
            momentum = 0.95, pooling = 1, cd_k = 5, 
            rho = 0.01, lambda_rate = 0.1, spmethod = 'entropy'):
     
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
        self.spmethod = spmethod
        if self.spmethod == 'relu':
            self.gradientSparsityConstraint = \
                self.gradientSparsityConstraintReLU
        else:
            self.gradientSparsityConstraint = \
                self.gradientSparsityConstraintEntropy

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

        self.compileTheanoFunctions()

    def saveModel(self, _filename):
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
        joblib.dump(pickleObject, _filename, protocol= 2)

    @classmethod
    def loadModel(cls, filename):

        numpyParams, hyperparams =joblib.load(filename)
        
        (num_motifs, motif_length, input_dims, \
            doublestranded, batchsize, learning_rate, \
            momentum, rho, lambda_rate,
            pooling, cd_k, 
            epochs, spmethod) = hyperparams

        obj = cls(num_motifs, motif_length, epochs, input_dims, \
                doublestranded, batchsize, learning_rate, \
                momentum, pooling, cd_k,
                rho, lambda_rate, spmethod)

        motifs, bias, c = numpyParams
        obj.motifs.set_value(motifs)
        obj.bias.set_value(bias)
        obj.c.set_value(c)
        return obj

    def bottomUpActivity(self, data, flip_motif=False):
        out = conv(data, self.motifs, filter_flip=flip_motif)
        out = out + self.bias.dimshuffle('x', 1, 0, 'x')
        return out

    def bottomUpProbability(self,activities):
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
        
    def bottomUpSample(self,probs):
        pool = self.pooling
        _probs=probs.reshape((probs.shape[0], probs.shape[1], probs.shape[2], probs.shape[3]//pool, pool))
        _probs_reshape=_probs.reshape((_probs.shape[0]*_probs.shape[1]*_probs.shape[2]*_probs.shape[3],pool))
        samples=self.theano_rng.multinomial(pvals=_probs_reshape)
        samples=samples.reshape((probs.shape[0],probs.shape[1],probs.shape[2],probs.shape[3]))
        return T.cast(samples,theano.config.floatX)

    def computeHgivenV(self, data, flip_motif=False):
        activity=self.bottomUpActivity(data,flip_motif)
        probability=self.bottomUpProbability(activity)
        sample=self.bottomUpSample(probability)
        return [probability, sample]


    def computeVgivenH(self, H_sample, H_sample_prime, softmaxdown=True):
        W = self.motifs.dimshuffle(1, 0, 2, 3)
        C = conv(H_sample, W, border_mode='full', filter_flip=True)
        
        out = T.sum(C, axis=1, keepdims=True)  # sum over all K

        if H_sample_prime:
            C = conv(H_sample_prime, W[:,:,::-1,::-1], \
                    border_mode='full', filter_flip=True)
            out = out+ T.sum(C, axis=1, keepdims=True)  # sum over all K

        c_bc = self.c
        c_bc = c_bc.dimshuffle('x', 0, 1, 'x')
        out = out + c_bc
        if softmaxdown:
          prob_of_V = self.softmax(out)
          # now, we still need the sample of V. Compute it here
          pV_ = prob_of_V.dimshuffle(0, 1, 3, 2).reshape( \
             (prob_of_V.shape[0]*prob_of_V.shape[3], prob_of_V.shape[2]))
          V_ = self.theano_rng.multinomial(n=1, 
               pvals=pV_).astype(theano.config.floatX)
          V = V_.reshape((prob_of_V.shape[0], 
              1, prob_of_V.shape[3], 
              prob_of_V.shape[2])).dimshuffle(0, 1, 3, 2)
        else:
          prob_of_V = self.sigmoid(out)
          V=self.theano_rng.multinomial(n=1,\
             pvals=prob_of_V).astype(theano.config.floatX)

        return [prob_of_V, V]

    def collectVHStatistics(self, prob_of_H, data):
        # reshape input
        data = data.dimshuffle(1, 0, 2, 3)
        prob_of_H = prob_of_H.dimshuffle(1, 0, 2, 3)
        avh = conv(data, prob_of_H, border_mode="valid", filter_flip=False)
        avh = avh / T.prod(prob_of_H.shape[1:])
        avh = avh.dimshuffle(1, 0, 2, 3).astype(theano.config.floatX)

        return avh

    def collectVStatistics(self, data):
        # reshape input
        a = T.mean(data, axis=(0, 1, 3)).astype(theano.config.floatX)
        a = a.dimshuffle('x', 0)
        a = T.inc_subtensor(a[:, :], a[:, ::-1])  # match a-t and c-g occurances

        return a

    def collectHStatistics(self, data):
        # reshape input
        a = T.mean(data, axis=(0, 2, 3)).astype(theano.config.floatX)
        a = a.dimshuffle('x', 0)

        return a

    def collectUpdateStatistics(self, prob_of_H, prob_of_H_prime, data):
        average_VH = self.collectVHStatistics(prob_of_H, data)
        average_H = self.collectHStatistics(prob_of_H)

        if prob_of_H_prime:
            average_VH_prime=self.collectVHStatistics(prob_of_H_prime, data)
            average_H_prime=self.collectHStatistics(prob_of_H_prime)
            average_VH=(average_VH+average_VH_prime[:,:,::-1,::-1])/2.
            average_H=(average_H+average_H_prime)/2.

        average_V = self.collectVStatistics(data)
        return average_VH, average_H, average_V
    
    def updateWeightsOnMinibatch(self, D, gibbs_chain_length):
        # calculate the data gradient for weights (motifs), bias and c
        [prob_of_H_given_data,H_given_data] = self.computeHgivenV(D)

        if self.doublestranded:
            [prob_of_H_given_data_prime,H_given_data_prime] = \
                    self.computeHgivenV(D, True)
        else:
            [prob_of_H_given_data_prime,H_given_data_prime] = [None, None]

        # calculate data gradients
        G_motif_data, G_bias_data, G_c_data = \
                  self.collectUpdateStatistics(prob_of_H_given_data, \
                  prob_of_H_given_data_prime, D)

        # calculate model probs
        H_given_model = self.fantasy_h
        if self.doublestranded:
            H_given_model_prime = self.fantasy_h_prime
        else:
            H_given_model_prime = None

        for i in range(gibbs_chain_length):
            prob_of_V_given_model, V_given_model = \
                    self.computeVgivenH(H_given_model, H_given_model_prime)
            #sample up
            prob_of_H_given_model, H_given_model = \
                    self.computeHgivenV(V_given_model)

            if self.doublestranded:
                prob_of_H_given_model_prime, H_given_model_prime = \
                        self.computeHgivenV(V_given_model,  True)
            else:
                prob_of_H_given_model_prime, H_given_model_prime = None, None
        
        # compute the model gradients
        G_motif_model, G_bias_model, G_c_model = \
                  self.collectUpdateStatistics(prob_of_H_given_model, \
                  prob_of_H_given_model_prime, V_given_model)
        
        mu = self.momentum
        alpha = self.learning_rate
        sp = self.lambda_rate
        reg_motif, reg_bias = self.gradientSparsityConstraint(D)

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

    def gradientSparsityConstraintReLU(self, data):
        # get expected[H|V]
        [prob_of_H, _] = self.computeHgivenV(data)
        gradKernels = T.grad(T.mean(T.nnet.relu(T.mean(prob_of_H, axis=(0, 2, 3)) -
                                                    self.rho)),
                             self.motifs)
        gradBias = T.grad(T.mean(T.nnet.relu(T.mean(prob_of_H, axis=(0, 2, 3)) -
                                                 self.rho)),
                          self.bias)
        return gradKernels, gradBias

    def gradientSparsityConstraintEntropy(self, data):
        # get expected[H|V]
        [prob_of_H, _] = self.computeHgivenV(data)
        q = self.rho
        p = T.mean(prob_of_H, axis=(0, 2, 3))

        gradKernels = - T.grad(T.mean(q*T.log(p) + (1-q)*T.log(1-p)),
                             self.motifs)
        gradBias = - T.grad(T.mean(q*T.log(p) + (1-q)*T.log(1-p)),
                          self.bias)
        return gradKernels, gradBias

    #def gradientSparsityConstraint(self, data):
        #return self.gradientSparsityConstraintReLU(data)
        #rdata = data
        #if self.hyper_params['sp_method'] == 'relu':
            #return self.gradientSparsityConstraintReLU(rdata)
        #elif self.hyper_params['sp_method'] == 'entropy':
            #return self.gradientSparsityConstraintEntropy(rdata)
        #else:
            #assert False, "sp_method '{}' not defined".format(\
                    #self.hyper_params['sp_method'])

    def compileTheanoFunctions(self):
        print "Start compiling Theano training function..."
        D = T.tensor4('data')
        updates = self.updateWeightsOnMinibatch(D, self.cd_k)
        self.trainingFun = theano.function(
              [D],
              None,
              updates=updates,
              name='train_CRBM'
        )

        #compute mean free energy
        mfe_ = self.meanFreeEnergy(D)
        #compute number  of motif hits
        [_, H] = self.computeHgivenV(D)
        
        #H = self.bottomUpProbability(self.bottomUpActivity(D))
        nmh_=T.mean(H)  # mean over samples (K x 1 x N_h)


        #compute norm of the motif parameters
        twn_=T.sqrt(T.mean(self.motifs**2))
        
        #compute information content
        pwm = self.softmax(self.motifs)
        entropy = -pwm * T.log2(pwm)
        entropy = T.sum(entropy, axis=2)  # sum over letters
        ic_= T.log2(self.motifs.shape[2]) - \
            T.mean(entropy)  # log is possible information due to length of sequence
        medic_= T.log2(self.motifs.shape[2]) - \
            T.mean(T.sort(entropy, axis=2)[:, :, entropy.shape[2] // 2])
        self.evaluateData = theano.function(
              [D],
              [mfe_, nmh_],
              name='evaluationData'
        )

        W=T.tensor4("W")
        self.evaluateParams = theano.function(
              [],
              [twn_,ic_,medic_],
                givens={W:self.motifs},
              name='evaluationParams'
        )
        fed=self.freeEnergyForData(D)
        self.freeEnergy=theano.function( [D],fed,name='fe_per_datapoint')

        fed=self.freeEnergyPerMotif(D)
        self.fePerMotif=theano.function( [D],fed,name='fe_per_motif')

        
        if self.doublestranded:
            Tfeat=T.mean(self.bottomUpActivity(D)+self.bottomUpActivity(D,True),axis=(2,3))
        else:
            Tfeat=T.mean(self.bottomUpActivity(D),axis=(2,3))
        self.featurize=theano.function([D],Tfeat)
        if self.doublestranded:
            Tfeat=T.mean(self.bottomUpActivity(D)+self.bottomUpActivity(D,True),axis=(2,3))
        else:
            Tfeat=T.mean(self.bottomUpProbability(self.bottomUpActivity(D)),axis=(2,3))

        if self.doublestranded:
            self.getHitProbs = theano.function([D], \
                self.bottomUpProbability(self.bottomUpActivity(D)))
        else:
            self.getHitProbs = theano.function([D], \
                #self.bottomUpProbability( T.maximum(self.bottomUpActivity(D),
                self.bottomUpProbability( self.bottomUpActivity(D) + 
                        self.bottomUpActivity(D, True)))
        print "Compilation of Theano training function finished"

    def fit(self, training_data, test_data):
        # assert that pooling can be done without rest to the division
        # compute sequence length
        nseq=int((training_data.shape[3]-\
            self.motif_length + 1)/\
            self.pooling)*\
            self.pooling+ \
            self.motif_length -1
        training_data=training_data[:,:,:,:nseq]
        nseq=int((test_data.shape[3]-\
            self.motif_length + 1)/\
            self.pooling)*\
            self.pooling+ \
            self.motif_length -1
        test_data=test_data[:,:,:,:nseq]

        # some debug printing
        numTrainingBatches = training_data.shape[0] / self.batchsize
        numTestBatches = test_data.shape[0] / self.batchsize
        print "BatchSize: " + str(self.batchsize)
        print "Num of iterations per epoch: " + str(numTrainingBatches)
        start = time.time()

        # compile training function

        # now perform training
        print "Start training the model..."
        starttime = time.time()

        for epoch in range(self.epochs):
            for [start,end] in self.iterateBatchIndices(\
                            training_data.shape[0],self.batchsize):
                self.trainingFun(training_data[start:end,:,:,:])
            meanfe=0.0
            meannmh=0.0
            nb=0
            for [start,end] in self.iterateBatchIndices(\
                            test_data.shape[0],self.batchsize):
                [mfe_,nmh_]=self.evaluateData(test_data[start:end,:,:,:])
                meanfe=meanfe+mfe_
                meannmh=meannmh+nmh_
                nb=nb+1
            [twn_,ic_,medic_]=self.evaluateParams()
            #for batchIdx in range(numTestBatches):
            print("Epoch {:d}: ".format(epoch) + \
                    "FE={:1.3f} ".format(meanfe/nb) + \
                    "NumH={:1.4f} ".format(meannmh/nb) + \
                    "WNorm={:2.2f} ".format(float(twn_)) + \
                    "IC={:1.3f} medIC={:1.3f}".format(float(ic_), float(medic_)))

        # done with training
        print "Training finished after: {:5.2f} seconds!".format(\
                time.time()-starttime)

    def meanFreeEnergy(self, D):
        return T.sum(self.freeEnergyForData(D))/D.shape[0]
        
    def getPFMs(self):
        def softmax_(x):
            x_exp = np.exp(x)
            y = np.zeros(x.shape)
            for i in range(x.shape[1]):
                y[:,i] = x_exp[:,i] / np.sum(x_exp[:,i])
            return y
        return [ softmax_(m[0, :, :]) for m in self.motifs.get_value() ]

    def freeEnergyForData(self, D):
        pool = self.pooling

        x=self.bottomUpActivity(D)

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3]//pool, pool))
        free_energy = -T.sum(T.log(1.+T.sum(T.exp(x), axis=4)), axis=(1, 2, 3))
        if self.doublestranded:
            x=self.bottomUpActivity(D,True)
  
            x = x.reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3]//pool, pool))
            free_energy = free_energy -T.sum(T.log(1.+T.sum(T.exp(x), axis=4)), axis=(1, 2, 3))
        
        cMod = self.c
        cMod = cMod.dimshuffle('x', 0, 1, 'x')  # make it 4D and broadcastable there
        free_energy = free_energy - T.sum(D * cMod, axis=(1, 2, 3))
        
        return free_energy/D.shape[3]

    def freeEnergyPerMotif(self, D):
        pool = self.pooling

        x=self.bottomUpActivity(D)

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3]//pool, pool))
        free_energy = -T.sum(T.log(1.+T.sum(T.exp(x), axis=4)), axis=(2, 3))

        if self.doublestranded:
            x=self.bottomUpActivity(D,True)
            x = x.reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3]//pool, pool))
            free_energy = free_energy -T.sum(T.log(1.+T.sum(T.exp(x), axis=4)), axis=(2, 3))
        
        cMod = self.c
        cMod = cMod.dimshuffle('x', 0, 1, 'x')  # make it 4D and broadcastable there
        free_energy = free_energy - T.sum(D * cMod, axis=(1, 2, 3)).dimshuffle(0, 'x')
        
        return free_energy

    def softmax(self, x):
        return T.exp(x) / T.exp(x).sum(axis=2, keepdims=True)

    def printHyperParams(self):
        print("num_motifs\t{:d}".format(self.num_motifs))
        print("motif_length\t {:d}".format(self.motif_length))
        print("input_dims\t {:d}".format(self.input_dims))
        print("doublestranded\t {}".format(self.doublestranded))
        print("batchsize\t {:d}".format(self.batchsize))
        print("learning_rate\t {:1.3f}".format(self.learning_rate))
        print("momentum\t {:1.3f}".format(self.momentum))
        print("rho\t\t {:1.4f}".format(self.rho))
        print("lambda_rate\t  {:1.3f}".format(self.lambda_rate))
        print("pooling\t  {:d}".format(self.pooling))
        print("cd_k\t\t   {:d}".format(self.cd_k))
        print("epochs\t  {}".format(self.epochs))

    def iterateBatchIndices(self, totalsize,nbatchsize):
        return [ [i,i+nbatchsize] if i+nbatchsize<=totalsize \
                    else [i,totalsize] for i in range(totalsize)[0::nbatchsize] ]
    
