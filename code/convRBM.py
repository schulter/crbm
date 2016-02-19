
# Theano imports
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d as conv
#from theano.sandbox.cuda.dnn import dnn_conv
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
        if self.hyper_params["doublestranded"]:
            b = np.zeros((1, 2*self.hyper_params['number_of_motifs'])).astype(theano.config.floatX)
        else:
            b = np.zeros((1, self.hyper_params['number_of_motifs'])).astype(theano.config.floatX)
        
        b = b - 7.
        c = np.zeros((1, 4)).astype(theano.config.floatX)

        self.bias = theano.shared(value=b, name='bias', borrow=True)
        self.c = theano.shared(value=c, name='c', borrow=True)

        # infrastructural parameters
        self.theano_rng = RS(seed=1234)
        self.params = [self.motifs, self.bias, self.c]
        
        self.debug = self.hyper_params['verbose']
        self.observers = []
        self.motif_velocity=theano.shared(value=np.zeros(self.motifs.get_value().shape).astype(theano.config.floatX), name='velocity_of_W',borrow=True)
        self.bias_velocity=theano.shared(value=np.zeros(b.shape).astype(theano.config.floatX), name='velocity_of_bias',borrow=True)
        self.c_velocity=theano.shared(value=np.zeros(c.shape).astype(theano.config.floatX), name='velocity_of_c',borrow=True)
        
        K = self.hyper_params['number_of_motifs']
        if self.hyper_params['doublestranded']:
            K *= 2
        
        if self.hyper_params['cd_method'] == 'pcd':
            self.fantasy_h = theano.shared(value=np.zeros((self.hyper_params['batch_size'], K, 1, 200)).astype(theano.config.floatX), name='fantasy_h', borrow=True)


    def initializeMotifs (self):

        if self.hyper_params["doublestranded"]:
            x = np.random.randn(2 * self.hyper_params['number_of_motifs'], 1, 4, self.hyper_params['motif_length']).astype(theano.config.floatX)
            # create reverse complement
            for i in range(0, 2*self.hyper_params['number_of_motifs'], 2):
                x[i+1] = x[i,:,::-1,::-1]
        else:
            x = np.random.randn(self.hyper_params['number_of_motifs'], 1, 4, self.hyper_params['motif_length']).astype(theano.config.floatX)
        
        self.motifs = theano.shared(value=x, name='W', borrow=True)


    def setCustomKernels (self, customKernels):
        assert (True), "doublestranded option not yet implemented"
        if len(customKernels.shape) != 4 or customKernels.shape[1] != 1:
            print "New motifs must be a 4D matrix with dims: (K x 1 x numOfLetters(4) x numOfKMers)"
            return

        if self.hyper_params['doublestranded']:
            assert (customKernels.shape[0] % 2 == 0), "doublestranded option with odd number of kernels not permitted"
            self.hyper_params['number_of_motifs'] = customKernels.shape[0] / 2
        else:
            self.hyper_params['number_of_motifs'] = customKernels.shape[0]

        self.hyper_params['motif_length'] = customKernels.shape[3]
        numMotifs = self.hyper_params['number_of_motifs']
        
        self.bias = theano.shared(value=b, name='bias', borrow=True)
        self.c = theano.shared(value=c, name='c', borrow=True)
        self.motifs = theano.shared(value=customKernels.astype(theano.config.floatX))
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
        out = conv(data, self.motifs, filter_flip=False)
        if self.debug:
            out = theano.printing.Print('Convolution result forward: ')(out)
        bMod = self.bias
        bMod = bMod.dimshuffle('x', 1, 0, 'x') # add dims to the bias until it works
        out = out + bMod
        
        # perform prob. max pooling g(filter(D,W) + b) and sampling
        if self.hyper_params['doublestranded']:
            pooled = max_pool(out.dimshuffle(0,2,1,3),
                pool_shape=(2, self.hyper_params['pooling_factor']),
                theano_rng=self.theano_rng)
        else:
            pooled = max_pool(out.dimshuffle(0,2,1,3),
                pool_shape=(1, self.hyper_params['pooling_factor']),
                theano_rng=self.theano_rng)

        prob_of_H = pooled[1].dimshuffle(0, 2, 1, 3)
        H = pooled[3].dimshuffle(0, 2, 1, 3)
        if self.debug:
            prob_of_H = theano.printing.Print('Hidden Probabilites: ')(prob_of_H)
            H = theano.printing.Print('prob max pooled layer: ')(H)
        return [prob_of_H,H] #only return pooled layer and probs


    def computeVgivenH (self, H_sample):

        W = self.motifs.dimshuffle(1, 0, 2, 3)
        C = conv(H_sample, W, border_mode='full',filter_flip=True)
        
        if self.debug:
            C = theano.printing.Print('Pre sigmoid visible layer: ')(C)
        out = T.sum(C, axis=1, keepdims=True) # sum over all K
        c_bc = self.c
        c_bc = c_bc.dimshuffle('x', 0, 1, 'x')
        out = out + c_bc
        prob_of_V = self.softmax(out)

        #return prob_of_V
        if self.debug:
            prob_of_V = theano.printing.Print('Softmax V (probabilities for V):')(prob_of_V)

        # now, we still need the sample of V. Compute it here
        pV_ = prob_of_V.dimshuffle(0, 1, 3, 2).reshape((prob_of_V.shape[0]*prob_of_V.shape[3], prob_of_V.shape[2]))
        V_ = self.theano_rng.multinomial(n=1,pvals=pV_).astype(theano.config.floatX)
        V = V_.reshape((prob_of_V.shape[0], 1, prob_of_V.shape[3], prob_of_V.shape[2])).dimshuffle(0, 1, 3, 2)
        if self.debug:
            V = theano.printing.Print('Visible Sample: ')(V)
        return [prob_of_V,V]

        
    def matchWeightchangeForComplementaryMotifs(self, evh,eh):

        evhre = evh.reshape((evh.shape[0]//2, 2, 1,evh.shape[2], evh.shape[3]))
        evhre_ = T.inc_subtensor(evhre[:,0,:,:,:], evhre[:,1,:,::-1,::-1])
        evhre = T.set_subtensor(evhre[:,1,:,:,:], evhre[:,0,:,::-1,::-1])
        evh=evhre.reshape(evh.shape)
        evh=evh/2.


        ehre = eh.reshape((1,eh.shape[1]//2, 2))
        ehre=T.inc_subtensor(ehre[:,:,0], ehre[:,:,1])
        ehre=T.set_subtensor(ehre[:,:,1], ehre[:,:,0])
        eh=ehre.reshape(eh.shape)
        eh=eh/2.

        return evh,eh
        
    def collectVHStatistics(self, prob_of_H, data):
        #reshape input 
        data=data.dimshuffle(1,0,2,3)
        prob_of_H=prob_of_H.dimshuffle(1,0,2,3)
        avh=conv(data,prob_of_H,border_mode="valid", filter_flip=False)
        avh=avh/ T.prod(prob_of_H.shape[1:])
        avh=avh.dimshuffle(1,0,2,3).astype(theano.config.floatX)

        return avh

    def collectVStatistics(self, data):
        #reshape input 
        a=T.mean(data,axis=(0,1,3)).astype(theano.config.floatX)
        a=a.dimshuffle('x',0)
        a=T.inc_subtensor(a[:,:],a[:,::-1]) #match a-t and c-g occurances

        return a

    def collectHStatistics(self, data):
    	  #reshape input 
        a=T.mean(data,axis=(0,2,3)).astype(theano.config.floatX)
        a=a.dimshuffle('x',0)

        return a

    def collectUpdateStatistics(self, prob_of_H, data):
        average_VH=self.collectVHStatistics(prob_of_H, data)
        average_H=self.collectHStatistics(prob_of_H)
        average_V=self.collectVStatistics(data)

        return average_VH, average_H, average_V

    
    def updateWeightsOnMinibatch (self, D, gibbs_chain_length):
        # calculate the data gradient for weights (motifs), bias and c
        [prob_of_H_given_data, H_given_data] = self.computeHgivenV(D)
        if self.debug:
            prob_of_H_given_data = theano.printing.Print('Hidden Layer Probabilities: ')(prob_of_H_given_data)
        # calculate data gradients
        G_motif_data, G_bias_data, G_c_data = self.collectUpdateStatistics(prob_of_H_given_data, D)
        
        if self.debug:
            G_motif_data = theano.printing.Print('Gradient for motifs (data): ')(G_motif_data)

        # calculate model probs
        if self.hyper_params['cd_method'] == 'pcd':
            H_given_model = self.fantasy_h
        else:
            H_given_model = H_given_data

        for i in range(gibbs_chain_length):
            prob_of_V_given_model, V_given_model = self.computeVgivenH(H_given_model)
            prob_of_H_given_model, H_given_model = self.computeHgivenV(V_given_model)
        
        # compute the model gradients
        G_motif_model, G_bias_model, G_c_model = self.collectUpdateStatistics(prob_of_H_given_model, V_given_model)
        
        if self.debug:
            G_motif_model = theano.printing.Print('Gradient for motifs (model): ')(G_motif_model)

        mu=self.hyper_params['momentum']
        alpha=self.hyper_params['learning_rate']
        sp=self.hyper_params['sparsity']

        reg_motif,reg_bias = self.gradientSparsityConstraint(D)
        #if self.hyper_params['doublestranded']:
        #   reg_motif,reg_bias = self.matchWeightchangeForComplementaryMotifs(reg_motif,reg_bias)

        # update the parameters and apply sparsity
        vmotifs = mu * self.motif_velocity + alpha* (G_motif_data - G_motif_model-sp*reg_motif)
        #vbias = mu * self.bias_velocity + alpha * (G_bias_data - G_bias_model-sp*reg_bias)
        vbias = mu * self.bias_velocity + 0 * (G_bias_data - G_bias_model-sp*reg_bias)
        vc = mu*self.c_velocity + alpha* (G_c_data - G_c_model)

        new_motifs = self.motifs + vmotifs
        new_bias = self.bias + vbias
        new_c = self.c + vc

        if self.hyper_params['doublestranded']:
            vmotifs,vbias = self.matchWeightchangeForComplementaryMotifs(vmotifs,vbias)
            new_motifs,new_bias = self.matchWeightchangeForComplementaryMotifs(new_motifs,new_bias)
        
        if self.hyper_params['cd_method'] == 'pcd':
            updates = [(self.motifs, new_motifs), (self.bias, new_bias), (self.c, new_c),
                       (self.motif_velocity,vmotifs),(self.bias_velocity,vbias),(self.c_velocity,vc),
                       (self.fantasy_h, H_given_model)]
        else:
            updates = [(self.motifs, new_motifs), (self.bias, new_bias), (self.c, new_c),
                       (self.motif_velocity,vmotifs),(self.bias_velocity,vbias),(self.c_velocity,vc)]

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
        updates = self.updateWeightsOnMinibatch(D,self.hyper_params['cd_k'])
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
					print str(obs.name) +": " + str(obs.calculateScore())
        for epoch in range(epochs):
            for batchIdx in range(itPerEpoch):
                trainingFun(batchIdx)
            for obs in self.observers:
							print str(obs.name) +": "+ str(obs.calculateScore())
            print "[Epoch " + str(epoch) + "] done!"

        # done with training
        print "Training finished after: " + str(time.time()-start) + " seconds!"


    def gradientSparsityConstraint(self, data):
        #get expected[H|V]
        prob_of_H, H=self.computeHgivenV(data)
        gradKernels = T.grad(T.mean(T.nnet.softplus(T.mean(prob_of_H,axis=(0,2,3))-self.hyper_params['rho'])), self.motifs)
        gradBias = T.grad(T.mean(T.nnet.softplus(T.mean(prob_of_H,axis=(0,2,3))-self.hyper_params['rho'])), self.bias)
        return gradKernels, gradBias

    def meanFreeEnergy (self, D):
        free_energy=0.0;

        x = conv(D, self.motifs, filter_flip=False)
        bMod = self.bias # to prevent member from being shuffled
        bMod = bMod.dimshuffle('x', 1, 0, 'x') # add dims to the bias on both sides
        x=x+bMod
        pool=self.hyper_params['pooling_factor']

        if self.hyper_params['doublestranded']:
            x=x.reshape((x.shape[0],x.shape[1]//2,2,x.shape[2],x.shape[3]//pool, pool))
            free_energy=free_energy - T.sum(T.log(1.+T.sum(T.exp(x),axis=(2,5))))
        else:
            x=x.reshape((x.shape[0],x.shape[1],x.shape[2],x.shape[3]//pool, pool))
            free_energy=free_energy - T.sum(T.log(1.+T.sum(T.exp(x),axis=(4))))
        
        cMod = self.c
        cMod = cMod.dimshuffle('x', 0, 1, 'x') # make it 4D and broadcastable there
        free_energy = free_energy - T.sum(D * cMod) 
        
        return free_energy/(D.shape[0]*D.shape[3])
        
        
    def softmax (self, x):
        return T.exp(x) / T.exp(x).sum(axis=2, keepdims=True)
        
    
    def printHyperParams (self):
        pprint.pprint(self.hyper_params)
