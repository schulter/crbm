# SNAIL


# Theano imports
import theano
import theano.tensor as T
import theano.tensor.nnet.conv as conv
#import theano.sandbox.cuda.dnn.dnn_conv as dnn_conv
from theano.sandbox.rng_mrg import MRG_RandomStreams as RS
from theano import pp

# numpy and python classics
import numpy as np
import random
import time
import cPickle
import pprint

from utils import max_pool



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
    def __init__ (self, hyperParams=None, file_name=None):
        
        if file_name == None and hyperParams == None:
            raise ArgumentError('Must Specify Either Filename or Hyper parameters')
            return
        
        if hyperParams == None:
            self.loadModel(file_name)
            return

        # parameters for the motifs
        self.hyper_params = hyperParams
        self.initializeMotifs()
        
        # cRBM parameters (2*x to respect both strands of the DNA)
        b = np.random.rand(1, 2*self.hyper_params['number_of_motifs']).astype(np.float32)
        c = np.random.rand(1, 4).astype(np.float32)
        self.bias = theano.shared(value=b, name='bias', borrow=True)
        self.c = theano.shared(value=c, name='c', borrow=True)

        # infrastructural parameters
        self.theano_rng = RS(seed=1234)
        self.params = [self.motifs, self.bias, self.c]
        
        self.debug = False
        self.setToZero = False
        self.observers = []


    def initializeMotifs (self):
        # create random motifs (2*self.numMotifs to respect both strands)
        x = np.random.rand(2 * self.hyper_params['number_of_motifs'], 1, 4, self.hyper_params['motif_length']).astype(np.float32)
        # create reverse complement
        for i in range(0, 2*self.hyper_params['number_of_motifs'], 2):
            x[i+1] = x[i,:,::-1,::-1]
        
        self.motifs = theano.shared(value=x, name='W', borrow=True)


    def setCustomKernels (self, customKernels):
        if len(customKernels.shape) != 4 or customKernels.shape[1] != 1:
            print "New motifs must be a 4D matrix with dims: (K x 1 x numOfLetters(4) x numOfKMers)"
            return

		
        self.hyper_params['number_of_motifs'] = (customKernels.shape[0] / 2)
        self.hyper_params['motif_length'] = customKernels.shape[3]
        numMotifs = self.hyper_params['number_of_motifs']
        
        if self.setToZero:
            b = np.zeros((1, 2*numMotifs)).astype(np.float32)
            c = np.zeros((1, 4)).astype(np.float32)
        else:
            b = np.random.rand(1, 2*numMotifs).astype(np.float32)
            c = np.random.rand(1, 4).astype(np.float32)

        self.bias = theano.shared(value=b, name='bias', borrow=True)
        self.c = theano.shared(value=c, name='c', borrow=True)
        self.motifs = theano.shared(value=customKernels.astype(np.float32))
        self.params = [self.motifs, self.bias, self.c]
        print "New motifs set. # Motifs: " + str(numMotifs) + " K-mer-Length: " + str(self.hyper_params['motif_length'])
        
    def addObserver (self, _observer):
        self.observers.append(_observer)

    def saveModel (self, _filename):
        numpyParams = []
        for param in self.params:
            numpyParams.append(param.get_value())
        
        pickleObject = (numpyParams, self.hyper_params, self.observers)
        with open(_filename, 'w') as f:
            cPickle.dump(pickleObject, f)


    def loadModel (self, filename):
        pickleObject = ()
        with open(filename, 'r') as f:
            pickleObject = cPickle.load(f)
        
        if pickleObject == ():
            raise IOError("Something went wrong loading the model!")
        
        numpyParams, self.hyper_params, self.observers = pickleObject
        
        # get the cRBM params done
        motifs, bias, c = numpyParams
        self.motifs = theano.shared(value=motifs, name='W', borrow=True)
        self.bias = theano.shared(value=bias, name='bias', borrow=True)
        self.c = theano.shared(value=c, name='c', borrow=True)
        self.params = [self.motifs, self.bias, self.c]
    


    def computeHgivenV (self, data):
        # calculate filter(D, W) + b
        out = conv.conv2d(data, self.motifs[:,:,::-1,::-1]) # cross-correlation
        #out = dnn_conv(data, self.motifs, conv_mode="cross") # cross-correlation
        if self.debug:
            out = theano.printing.Print('Convolution result forward: ')(out)
        bMod = self.bias
        bMod = bMod.dimshuffle('x', 1, 0, 'x') # add dims to the bias until it works
        out = out + bMod
        
        # perform prob. max pooling g(filter(D,W) + b) and sampling
        pooled = max_pool(out.dimshuffle(0,2,1,3),
						  pool_shape=(2, self.hyper_params['pooling_factor']),
						  theano_rng=self.theano_rng
						 )

        prob_of_H = pooled[1].dimshuffle(0, 2, 1, 3)
        H = pooled[3].dimshuffle(0, 2, 1, 3)
        if self.debug:
            prob_of_H = theano.printing.Print('Hidden Probabilites: ')(prob_of_H)
            H = theano.printing.Print('prob max pooled layer: ')(H)
        return [prob_of_H,H] #only return pooled layer and probs


    def computeVgivenH (self, H_sample):
        # first, compute P(V | H) via convolution
        # P(V | H) = softmax( conv(H, W) + c )
        # dimshuffle the motifs to have K as channels (will be summed automatically)
        W = self.motifs.dimshuffle(1, 0, 2, 3)
        C = conv.conv2d(H_sample, W, border_mode='full')
        if self.debug:
            C = theano.printing.Print('Pre sigmoid visible layer: ')(C)
        out = T.sum(C, axis=1, keepdims=True) # sum over all K
        c_bc = self.c
        c_bc = c_bc.dimshuffle('x', 0, 1, 'x')
        out = out + c_bc
        prob_of_V = self.softmax(out)

        if self.debug:
            prob_of_V = theano.printing.Print('Softmax V (probabilities for V):')(prob_of_V)

        # now, we still need the sample of V. Compute it here
        pV_ = prob_of_V.dimshuffle(0, 1, 3, 2).reshape((prob_of_V.shape[0]*prob_of_V.shape[3], prob_of_V.shape[2]))
        V_ = self.theano_rng.multinomial(n=1,pvals=pV_)
        V = V_.reshape((prob_of_V.shape[0], 1, prob_of_V.shape[3], prob_of_V.shape[2])).dimshuffle(0, 1, 3, 2)
        V = V.astype('float32')
        if self.debug:
            V = theano.printing.Print('Visible Sample: ')(V)
        return [prob_of_V,V]

        
    def makeDerivativesStrandCompliant (self, derivatives):
        # reshape such that even kernels form one matrix, while the uneven form the other
        N_batch, K, letters, length = derivatives.shape
        D_reshaped = derivatives.reshape((N_batch, K//2, 2, letters, length))
        
        # sum together the even and uneven ones and construct the reverse thing
        D_summed = D_reshaped[:,:,0,:,:] + D_reshaped[:,:,1,:,:]
        D_summed_reverse = D_summed[:,:,::-1,::-1] # just invert cols and rows of kernel
        
        # melt it all back together by first adding yet another dimension
        D_restored = T.stack(D_summed, D_summed_reverse)
        D_result = D_restored.dimshuffle(1, 2, 0, 3, 4).reshape((N_batch, K, letters, length))
        
        if self.debug:
            D_result = theano.printing.Print('Derivatives strand compliant')(D_result)

        return D_result
        

    def collectUpdateStatistics(self, hiddenProbs, data):
        
        # new code to capture 1 <-> 1 relationship
        #assert data.shape[0] == hiddenProbs.shape[0]
        N_batch = data.shape[0]
        C_data = T.zeros((N_batch, 2*self.hyper_params['number_of_motifs'], 4, self.hyper_params['motif_length']))
        
        for seq in range(self.batchSize):
            d_i = data[seq].dimshuffle('x',0,1,2)
            h_i = hiddenProbs[seq].dimshuffle(0,'x',1,2)[:,:,::-1,::-1]
            subT_result = C_data[seq]
            localResult = conv.conv2d(d_i, h_i).sum(axis=0)
            C_data = T.set_subtensor(subT_result, localResult)

        # make the kernels respect the strand structure
        #C_data = self.makeDerivativesStrandCompliant(C_data)

        der_Motifs = T.sum(C_data, axis=0, keepdims=True) / self.hyper_params['number_of_motifs'] # mean over training examples
        der_Motifs = der_Motifs.dimshuffle(1, 0, 2, 3) # bring back to former shape
        der_bias = T.mean(T.sum(hiddenProbs, axis=3), axis=0).dimshuffle(1,0)
        der_c = T.mean(T.sum(data, axis=3), axis=0)

        return (der_Motifs, der_bias, der_c)
    
    def updateWeightsOnMinibatch (self, D, numOfCDs):
        # calculate the data gradient for weights (motifs) and bias
        [prob_of_H_given_data, H_given_data] = self.computeHgivenV(D)
        if self.debug:
            prob_of_H_given_data = theano.printing.Print('Hidden Layer Probabilities: ')(prob_of_H_given_data)
        # calculate data gradients
        G_motif_data, G_bias_data, G_c_data = self.collectUpdateStatistics(prob_of_H_given_data, D)
        
        if self.debug:
            G_motif_data = theano.printing.Print('Gradient for motifs (data): ')(G_motif_data)
        # calculate model probs
        H_given_model = H_given_data
        for i in range(numOfCDs):
            prob_of_V_given_model, V_given_model = self.computeVgivenH(H_given_model)
            prob_of_H_given_model, H_given_model = self.computeHgivenV(V_given_model)
        
        # compute the model gradients
        G_motif_model, G_bias_model, G_c_model = self.collectUpdateStatistics(prob_of_H_given_model, V_given_model)
        
        if self.debug:
            G_motif_model = theano.printing.Print('Gradient for motifs (model): ')(G_motif_model)
        
        # update the parameters
        new_motifs = self.motifs + self.hyper_params['learning_rate'] * (G_motif_data - G_motif_model)
        new_bias = self.bias + self.hyper_params['learning_rate'] * (G_bias_data - G_bias_model)
        new_c = self.c + self.hyper_params['learning_rate'] * (G_c_data - G_c_model)
        
        #score = self.getDataReconstruction(D)
        updates = [(self.motifs, new_motifs), (self.bias, new_bias), (self.c, new_c)]

        return updates
    
    
    def trainModel (self, trainData):

        # assert that pooling can be done without rest to the division
        assert (((trainData.shape[3] - self.hyper_params['motif_length'] + 1) % self.hyper_params['pooling_factor']) == 0)

        batchSize=self.hyper_params['batch_size']
        epochs=self.hyper_params['epochs']
        # some debug printing
        itPerEpoch = trainData.shape[0] / batchSize
        print "BatchSize: " + str(batchSize)
        print "Num of iterations per epoch: " + str(itPerEpoch)
        start = time.time()

        # compile training function
        print "Start compiling Theano training function..."
        train_set = theano.shared(value=trainData, borrow=True, name='trainData')
        index = T.lscalar()
        D = T.tensor4('data')
        updates = self.updateWeightsOnMinibatch(D, self.hyper_params['cd_k'])
        trainingFun = theano.function(
              [index],
              None,
              updates = updates,
              allow_input_downcast=True,
              givens={
                D: train_set[index*batchSize:(index+1)*batchSize]
              },
              name='train_CRBM'
        )
        print "Compilation of Theano training function finished in " + str(time.time()-start) + " seconds"

        # now perform training
        print "Start training the model..."
        start = time.time()
        for obs in self.observers:
            print "Initial Score of " + str(obs.name) + ": " + str(obs.calculateScore())
        for epoch in range(epochs):
            for batchIdx in range(itPerEpoch):
                trainingFun(batchIdx)
            for obs in self.observers:
                print "Score of " + str(obs.name) + ": " + str(obs.calculateScore())
            print "[Epoch " + str(epoch) + "] done!"

        # done with training
        print "Training finished after: " + str(time.time()-start) + " seconds!"


    def getReconFun (self):
        D = T.tensor4('data')
        score = self.getDataReconstruction(D)
        return theano.function([D], score, allow_input_downcast=True)
    
    
    def getDataReconstruction (self, D):
        [prob_of_H, H] = self.computeHgivenV(D)
        [prob_of_V,V] = self.computeHgivenV(H)
        #S_V = self.sampleVisibleLayer(V)
        sames = V * D # elements are 1 if they have the same letter...
        count = T.sum(T.mean(sames, axis=0)) # mean over samples, sum over rest
        return count
    
 
    def getFreeEnergyFunction (self):
        D = T.tensor4('data')
        free_energy = self.calculateFreeEnergy(D)
        return theano.function([D], free_energy, allow_input_downcast=True)
    
    
    def calculateFreeEnergy (self, D):
        # firstly, compute hidden part of free energy
        C = conv.conv2d(D, self.motifs)
        bMod = self.bias # to prevent member from being shuffled
        bMod = bMod.dimshuffle('x', 1, 0, 'x') # add dims to the bias on both sides
        C = C + bMod
        hiddenPart = T.sum(T.log(1. + T.exp(C)), axis=1) # dim: N_batch x 1 x N_h after sum over K
        hiddenPart = T.sum(T.mean(hiddenPart, axis=0)) # mean over all samples and sum over units
        
        # compute the visible part
        cMod = self.c
        cMod = cMod.dimshuffle('x', 0, 1, 'x') # make it 4D and broadcastable there
        visiblePart = T.mean(D * cMod, axis=0) # dim: 1 x 4 x N_v
        visiblePart = T.sum(visiblePart)
        
        return hiddenPart + visiblePart # don't return the negative because it's more difficult to plot
        
        
    def softmax (self, x):
        return T.exp(x) / T.exp(x).sum(axis=2, keepdims=True)
        
    
    def printHyperParams (self):
        pprint.pprint(self.hyper_params)
