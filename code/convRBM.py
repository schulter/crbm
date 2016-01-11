# Theano imports
import theano
import theano.tensor as T
import theano.tensor.nnet.conv as conv
from theano.sandbox.rng_mrg import MRG_RandomStreams as RS
from theano import pp

# numpy and python classics
import numpy as np
import random
import time
import cPickle

from utils import max_pool

## PART 3: Optimizing theano to do it all on the GPU

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
    def __init__ (self, _motifLength, _numMotifs, _learningRate=0.1, _poolingFactor=1):
        # parameters for the motifs
        self.motifLength = _motifLength
        self.numMotifs = _numMotifs
        self.initializeMotifs()
        
        # cRBM parameters (2*x to respect both strands of the DNA)
        b = np.random.rand(1, 2*self.numMotifs).astype(np.float32)
        c = np.random.rand(1, 4).astype(np.float32)
        self.bias = theano.shared(value=b, name='bias', borrow=True)
        self.c = theano.shared(value=c, name='c', borrow=True)
        self.poolingFactor = _poolingFactor
        self.learningRate = _learningRate
        
        # infrastructural parameters
        self.theano_rng = RS(seed=1234)
        self.params = [self.motifs, self.bias, self.c]
        self.debug = False
        self.observers = []
    
    
    def initializeMotifs (self):
        # create random motifs (2*self.numMotifs to respect both strands)
        x = np.random.rand(2 * self.numMotifs, 1, 4, self.motifLength).astype(np.float32)
        
        # create reverse complement
        for i in range(0, 2*self.numMotifs, 2):
            x[i+1] = x[i,:,::-1,::-1]
            
        self.motifs = theano.shared(value=x, name='W', borrow=True)
        
        
    def setCustomKernels (self, customKernels):
        if len(customKernels.shape) != 4 or customKernels.shape[1] != 1:
            print "New motifs must be a 4D matrix with dims: (K x 1 x numOfLetters(4) x numOfKMers)"
            return
        
        self.numMotifs = (customKernels.shape[0] / 2)
        self.motifLength = customKernels.shape[3]
        #b = np.random.rand(1, self.numMotifs).astype(np.float32)
        
        if self.debug:
            b = np.zeros((1, 2*self.numMotifs)).astype(np.float32)
            c = np.zeros((1, 4)).astype(np.float32)
        else:
            b = np.random.rand(1, 2*self.numMotifs).astype(np.float32)
            c = np.random.rand(1, 4).astype(np.float32)

        self.bias = theano.shared(value=b, name='bias', borrow=True)
        self.c = theano.shared(value=c, name='c', borrow=True)
        
        self.motifs = theano.shared(value=customKernels.astype(np.float32))
        self.params = [self.motifs, self.bias, self.c]
        print "New motifs set. # Motifs: " + str(self.numMotifs) + " K-mer-Length: " + str(self.motifLength)

    def addObserver (self, _observer):
        self.observers.append(_observer)
        
        
    def saveModel (self, _filename):
        numpyParams = []
        for param in self.params:
            numpyParams.append(param.get_value())

        with open(_filename, 'w') as f:
            cPickle.dump(numpyParams, f)

    def loadModel (self, filename):
        numpyParams = []
        with open(filename, 'r') as f:
            numpyParams = cPickle.load(f)

        if numpyParams == []:
            raise IOError("Something went wrong loading the model!")
        motifs, bias, c = numpyParams
        self.motifs = theano.shared(value=motifs, name='W', borrow=True)
        self.bias = theano.shared(value=bias, name='bias', borrow=True)
        self.c = theano.shared(value=c, name='c', borrow=True)

        self.params = [self.motifs, self.bias, self.c]
        return
    
    
### ------------------------------THE TOUGH STUFF-------------------------------- ###
### ----------------------------------------------------------------------------- ###

    def forwardBatch (self, data):
        # calculate filter(D, W) + b
        out = conv.conv2d(data, self.motifs[:,:,::-1,::-1])
        if self.debug:
            out = theano.printing.Print('Convolution result forward: ')(out)
        bMod = self.bias
        bMod = bMod.dimshuffle('x', 1, 0, 'x') # add dims to the bias until it works
        out = out + bMod
        
        # perform prob. max pooling g(filter(D,W) + b) and sampling
        pooled = max_pool(out.dimshuffle(0,2,1,3), pool_shape=(2, self.poolingFactor), theano_rng=self.theano_rng)
        H = pooled[1].dimshuffle(0,2,1,3)
        S = pooled[3].dimshuffle(0,2,1,3)
        if self.debug:
            H = theano.printing.Print('Hidden Probabilites: ')(H)
            S = theano.printing.Print('prob max pooled layer: ')(S)
        return [H,S] #only return pooled layer and probs


    def backwardBatch (self, H_sample):
        # dimshuffle the motifs to have 
        W = self.motifs.dimshuffle(1, 0, 2, 3)[:,:,::-1,::-1] # kernel is flipped prior to convolution
        C = conv.conv2d(H_sample, W, border_mode='full')[:,:,::-1,:]
        if self.debug:
            C = theano.printing.Print('Pre sigmoid visible layer: ')(C)
        out = T.sum(C, axis=1, keepdims=True) # sum over all K
        c_bc = self.c
        c_bc = c_bc.dimshuffle('x', 0, 1, 'x')
        out = out + c_bc
        res = self.softmax(out)
        return res

        
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
        

    def expectedDerivative (self, hiddenProbs, data):
        
        # new code to capture 1 <-> 1 relationship
        #assert data.shape[0] == hiddenProbs.shape[0]
        N_batch = data.shape[0]
        result = T.zeros((N_batch, 2*self.numMotifs, 4, self.motifLength))
        
        for seq in range(self.batchSize):
            d_i = data[seq,:,:,:].dimshuffle('x',0,1,2)
            h_i = hiddenProbs[seq,:,:,:].dimshuffle(0,'x',1,2)
            subT_result = result[seq,:,:,:]
            localResult = conv.conv2d(d_i, h_i).sum(axis=0)
            result = T.set_subtensor(subT_result, localResult)

        # end of new code
        
        #mean = T.mean(hiddenProbs, axis=0, keepdims=True) # mean over H so only one mean datapoint
        #H_reshaped = mean.dimshuffle(1, 0, 2, 3)
        # TODO: Capture the 1 <-> 1 relation between samples in H and D
        # Currently, this is done by mean (1st row) but that's not good at all
        #out = conv.conv2d(data, H_reshaped)
        
        out = result
        
        # make the kernels respect the strand structure
        #out = self.makeDerivativesStrandCompliant(out)
        
        der_Motifs = T.sum(out, axis=0, keepdims=True) / self.numMotifs # mean over training examples
        der_Motifs = der_Motifs.dimshuffle(1, 0, 2, 3) # bring back to former shape
        der_bias = T.mean(T.sum(hiddenProbs, axis=3), axis=0).dimshuffle(1,0)
        der_c = T.mean(T.sum(data, axis=3), axis=0)
        return (der_Motifs, der_bias, der_c)
    
    
        
    def train_model (self, D, numOfCDs):
        # calculate the data gradient for weights (motifs) and bias
        [H_data, S_data] = self.forwardBatch(D)
        if self.debug:
            H_data = theano.printing.Print('Hidden Layer Probabilities: ')(H_data)
        # calculate data gradients
        G_motif_data, G_bias_data, G_c_data = self.expectedDerivative(H_data, D)
        
        if self.debug:
            G_motif_data = theano.printing.Print('Gradient for motifs (data): ')(G_motif_data)
        # calculate model probs
        S_H = S_data
        for i in range(numOfCDs):
            V_model = self.backwardBatch(S_H)
            S_V_model = self.sampleVisibleLayer(V_model)
            [H_model, S_H] = self.forwardBatch(S_V_model)
        
        # compute the model gradients
        G_motif_model, G_bias_model, G_c_model = self.expectedDerivative(H_model, D)
        
        if self.debug:
            G_motif_model = theano.printing.Print('Gradient for motifs (model): ')(G_motif_model)
        
        # update the parameters
        new_motifs = self.motifs + self.learningRate * (G_motif_data - G_motif_model)
        new_bias = self.bias + self.learningRate * (G_bias_data - G_bias_model)
        new_c = self.c + self.learningRate * (G_c_data - G_c_model)
        
        #score = self.getDataReconstruction(D)
        updates = [(self.motifs, new_motifs), (self.bias, new_bias), (self.c, new_c)]

        return updates
    
    
    def trainMinibatch (self, trainData, testData, epochs, batchSize, numOfCDs):

        # assert that pooling can be done without rest to the division
        assert (((trainData.shape[3] - self.motifLength + 1) % self.poolingFactor) == 0)
        assert (((testData.shape[3] - self.motifLength + 1) % self.poolingFactor) == 0)

        self.batchSize = batchSize
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
        updates = self.train_model(D, numOfCDs)
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

        for epoch in range(epochs):
            for batchIdx in range(itPerEpoch):
                trainingFun(batchIdx)
            for obs in self.observers:
                print "Score of function: " + str(obs.calculateScore())
            print "[Epoch " + str(epoch) + "] done!"

        # done with training
        print "Training finished after: " + str(time.time()-start) + " seconds!"


    def getReconFun (self):
        D = T.tensor4('data')
        score = self.getDataReconstruction(D)
        return theano.function([D], score, allow_input_downcast=True)
    
    
    def getDataReconstruction (self, D):
        [H, S_H] = self.forwardBatch(D)
        V = self.backwardBatch(S_H)
        S_V = self.sampleVisibleLayer(V)
        sames = S_V * D # elements are 1 if they have the same letter...
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
        
        
    def sampleVisibleLayer (self, V):
        reshaped = V.dimshuffle(0, 1, 3, 2).reshape((V.shape[0]*V.shape[3], V.shape[2]))
        S_reshaped = self.theano_rng.multinomial(n=1,pvals=reshaped)
        S = S_reshaped.reshape((V.shape[0], 1, V.shape[3], V.shape[2])).dimshuffle(0, 1, 3, 2)
        S = S.astype('float32')
        if self.debug:
            S = theano.printing.Print('Visible Sample: ')(S)
        return S
    
    def softmax (self, x):
        return T.exp(x) / T.exp(x).sum(axis=2, keepdims=True)
        
