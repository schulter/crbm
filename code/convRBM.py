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

        
### ------------------------------THE TOUGH STUFF-------------------------------- ###
### ----------------------------------------------------------------------------- ###

    def forwardBatch (self, data):
        out = conv.conv2d(data, self.motifs[:,:,::-1,::-1])
        if self.debug:
            out = theano.printing.Print('Convolution result forward: ')(out)
        bMod = self.bias
        bMod = bMod.dimshuffle('x', 1, 0, 'x') # add dims to the bias until it works
        out = out + bMod
        pooled = max_pool(out.dimshuffle(0,2,1,3), pool_shape=(2, self.poolingFactor), theano_rng=self.theano_rng)
        H = pooled[1]
        S = pooled[3]
        if self.debug:
            H = theano.printing.Print('Hidden Probabilites: ')(H)
            S = theano.printing.Print('prob max pooled layer: ')(S)
        return [H,S] #only return pooled layer and probs


    def backwardBatch (self, H_sample):
        K = self.motifs.dimshuffle(1, 0, 2, 3)[:,:,::-1,::-1] # kernel is flipped prior to convolution
        H_shuffled = H_sample.dimshuffle(0, 2, 1, 3) # interpret the kernels as channels (will be summed automatically)
        C = conv.conv2d(H_shuffled, K, border_mode='full')[:,:,::-1,:]
        if self.debug:
            C = theano.printing.Print('Pre sigmoid visible layer: ')(C)
        out = T.sum(C, axis=1, keepdims=True) # sum over all K
        c_bc = self.c
        c_bc = c_bc.dimshuffle('x', 0, 1, 'x')
        out = out + c_bc
        res = self.softmax(out)
        return res

    
    def makeDerivativesStrandCompliant (self, derivatives):
        # scan over the even kernels and sum them together with the next
        # one (k[0] + k[1] etc)
        def sumWithNext(idx, output_model):
            sub1 = output_model[:,idx,:,:]
            sub2 = output_model[:,idx+1,:,:]
            added = sub1 + sub2
            out = T.set_subtensor(sub1, added)
            return out
        result, updates = theano.scan(fn=sumWithNext,
                                      outputs_info=derivatives,
                                      sequences=[T.arange(start=0, stop=2*self.numMotifs, step=2)])
        result = result[-1] # only take the last one
        
        # now scan over the unevens and set their derivative to the reverse complement
        # of the evens
        def setToReverseComplement(idx, output_model):
            sub1 = output_model[:,idx,:,:]
            revCom = output_model[:,idx-1,:,:]
            revCom = revCom[:,::-1,::-1]
            return T.set_subtensor(sub1, revCom)
        result, updates = theano.scan(fn=setToReverseComplement,
                                      outputs_info=result,
                                      sequences=[T.arange(start=1, stop=2*self.numMotifs, step=2)])
        
        return result[-1]
        

    def expectedDerivative (self, hiddenProbs, data):
        mean = T.mean(hiddenProbs, axis=0, keepdims=True) # sum over all training data to get avg (but keep dim)
        H_reshaped = mean.dimshuffle(2, 0, 1, 3)
        # TODO: Capture the 1 <-> 1 relation between samples in H and D
        # Currently, this is done by mean (1st row) but that's not good at all
        out = conv.conv2d(data, H_reshaped)
        # TODO: perform scan here and sum even ones while setting unevens to 0.
        # TODO: Don't mean over all then, but use sum and divide by K (not 2*K like in this case)
        out = self.makeDerivativesStrandCompliant(out)
        
        der_Motifs = T.sum(out, axis=0, keepdims=True) / self.numMotifs # mean over training examples
        der_Motifs = der_Motifs.dimshuffle(1, 0, 2, 3) # bring back to former shape
        der_bias = T.mean(T.sum(hiddenProbs, axis=3), axis=0)
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
        
        # some debug printing
        itPerEpoch = trainData.shape[0] / batchSize
        print "BatchSize: " + str(batchSize)
        print "Num of iterations per epoch: " + str(itPerEpoch)
        start = time.time()
        
        # compile training function
        
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

        reconFun = self.getFreeEnergyFunction()
        print "Compilation of theano function finished in " + str(time.time()-start) + " seconds"
        print "Start training..."
        start = time.time()
        allScores = []
        allScores.append(reconFun(testData))
        print "Initial Reconstruction Error: " + str(allScores[-1])
        for epoch in range(epochs):
            smallScores = []
            for batchIdx in range(itPerEpoch):
                trainingFun(batchIdx)
            allScores.append(reconFun(testData))
            print "[Epoch " + str(epoch) + "] Reconstruction Error: " + str(allScores[-1])
        print "Training finished after: " + str(time.time()-start) + " seconds!"
        return allScores

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
        return score
    
 
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
        
