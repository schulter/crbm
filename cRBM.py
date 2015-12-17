

# Theano imports
import theano
import theano.tensor as T
import theano.tensor.nnet.conv as conv
from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RS
from theano import pp
from theano.gof.op import get_debug_values

# numpy and python classics
import numpy as np
import random
import time

# biopython stuff
import Bio.SeqIO as sio
import Bio.motifs.matrix as mat
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio import motifs

# plotting
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split



"""
This class reads sequences from fasta files.
To use it, create an instance of that object and use
the function readSequencesFromFile.
"""
class FASTAReader:
    
    def __init__(self, _path):
        self.path = _path
        
    def readSequencesFromFile (self, filename):
        dhsSequences = []
        for dhs in sio.parse(open(filename), 'fasta', IUPAC.unambiguous_dna):
            dhsSequences.append(dhs.seq)
        return dhsSequences
    
    
class JASPARReader:
    
    def __init__ (self):
        pass
    
    def readSequencesFromFile (self, filename):
        matrices = []
        for mat in motifs.parse(open(filename), 'jaspar'):
            matrices.append(mat.pwm)
        return matrices
        

def getIntToLetter (letter):
    if letter == 'A' or letter == 'a':
        return 0
    elif letter == 'C' or letter == 'c':
        return 1
    elif letter == 'G' or letter == 'g':
        return 2
    elif letter == 'T' or letter == 't':
        return 3
    else:
        print "ERROR. LETTER " + letter + " DOES NOT EXIST!"
        return -1

def getMatrixFromSeq (seq):
    m = len(seq.alphabet.letters)
    n = len(seq)
    result = np.zeros((1, m, n), dtype=np.float32)
    revSeq = seq.reverse_complement()
    for i in range(len(seq)):
        result[0,getIntToLetter(seq[i]),i] = 1
    return result




def max_pool(z, pool_shape, top_down=None, theano_rng=None):
    """
    Probabilistic max-pooling
    Parameters
    ----------
    z : theano 4-tensor
        a theano 4-tensor representing input from below
    pool_shape : tuple
        tuple of ints. the shape of regions to be pooled
    top_down : theano 4-tensor, optional
        a theano 4-tensor representing input from above
        if None, assumes top-down input is 0
    theano_rng : MRG_RandomStreams, optional
        Used for random numbers for sampling
    Returns
    -------
    p : theano 4-tensor
        the expected value of the pooling layer p
    h : theano 4-tensor
        the expected value of the detector layer h
    p_samples : theano 4-tensor, only returned if theano_rng is not None
        samples of the pooling layer
    h_samples : theano 4-tensor, only returned if theano_rng is not None
        samples of the detector layer
    Notes
    ------
    all 4-tensors are formatted with axes ('b', 'c', 0, 1).
    This is for maximum speed when using theano's conv2d
    to generate z and top_down, or when using it to infer conditionals of
    other layers using the return values.
    Detailed description:
    Suppose you have a variable h that lives in a Conv2DSpace h_space and
    you want to pool it down to a variable p that lives in a smaller
    Conv2DSpace p.
    This function does that, using non-overlapping pools.
    Specifically, consider one channel of h. h must have a height that is a
    multiple of pool_shape[0] and a width that is a multiple of pool_shape[1].
    A channel of h can thus be broken down into non-overlapping rectangles
    of shape pool_shape.
    Now consider one rectangular pooled region within one channel of h.
    I now use 'h' to refer just to this rectangle, and 'p' to refer to
    just the one pooling unit associated with that rectangle.
    We assume that the space that h and p live in is constrained such
    that h and p are both binary and p = max(h). To reduce the state-space
    in order to make probabilistic computations cheaper we also
    constrain sum(h) <= 1.
    Suppose h contains k different units. Suppose that the only term
    in the model's energy function involving h is -(z*h).sum()
    (elemwise multiplication) and the only term in
    the model's energy function involving p is -(top_down*p).sum().
    Then P(h[i] = 1) = softmax( [ z[1], z[2], ..., z[k], -top_down] )[i]
    and P(p = 1) = 1-softmax( [z[1], z[2], ..., z[k], -top_down])[k]
    This variation of the function assumes that z, top_down, and all
    return values use Conv2D axes ('b', 'c', 0, 1).
    This variation of the function implements the softmax using a
    theano graph of exp, maximum, sub, and div operations.
    Performance notes:
    It might be possible to make a faster implementation with different
    theano ops. rather than using set_subtensor, it might be possible
    to use the stuff in theano.sandbox.neighbours. Probably not possible,
    or at least nasty, because that code isn't written with multiple
    channels in mind, and I don't think just a reshape can fix it.
    Some work on this in galatea.cond.neighbs.py
    At some point images2neighbs' gradient was broken so check that
    it has been fixed before sinking too much time into this.
    Stabilizing the softmax is also another source of slowness.
    Here it is stabilized with several calls to maximum and sub.
    It might also be possible to stabilize it with
    T.maximum(-top_down,T.signal.downsample.max_pool(z)).
    Don't know if that would be faster or slower.
    Elsewhere in this file I implemented the softmax with a reshape
    and call to Softmax / SoftmaxWithBias.
    This is slower, even though Softmax is faster on the GPU than the
    equivalent max/sub/exp/div graph. Maybe the reshape is too expensive.
    Benchmarks show that most of the time is spent in GpuIncSubtensor
    when running on gpu. So it is mostly that which needs a faster
    implementation. One other way to implement this would be with
    a linear.Conv2D.lmul_T, where the convolution stride is equal to
    the pool width, and the thing to multiply with is the hparts stacked
    along the channel axis. Unfortunately, conv2D doesn't work right
    with stride > 2 and is pretty slow for stride 2. Conv3D is used to
    mitigate some of this, but only has CPU code.
    """

    z_name = z.name
    if z_name is None:
        z_name = 'anon_z'

    batch_size, ch, zr, zc = z.shape

    r, c = pool_shape

    zpart = []

    mx = None

    if top_down is None:
        t = 0.
    else:
        t = - top_down
        t.name = 'neg_top_down'

    for i in xrange(r):
        zpart.append([])
        for j in xrange(c):
            cur_part = z[:, :, i:zr:r, j:zc:c]
            if z_name is not None:
                cur_part.name = z_name + '[%d,%d]' % (i, j)
            zpart[i].append(cur_part)
            if mx is None:
                mx = T.maximum(t, cur_part)
                if cur_part.name is not None:
                    mx.name = 'max(-top_down,' + cur_part.name + ')'
            else:
                max_name = None
                if cur_part.name is not None:
                    mx_name = 'max(' + cur_part.name + ',' + mx.name + ')'
                mx = T.maximum(mx, cur_part)
                mx.name = mx_name
    mx.name = 'local_max(' + z_name + ')'

    pt = []

    for i in xrange(r):
        pt.append([])
        for j in xrange(c):
            z_ij = zpart[i][j]
            safe = z_ij - mx
            safe.name = 'safe_z(%s)' % z_ij.name
            cur_pt = T.exp(safe)
            cur_pt.name = 'pt(%s)' % z_ij.name
            pt[-1].append(cur_pt)

    off_pt = T.exp(t - mx)
    off_pt.name = 'p_tilde_off(%s)' % z_name
    denom = off_pt

    for i in xrange(r):
        for j in xrange(c):
            denom = denom + pt[i][j]
    denom.name = 'denom(%s)' % z_name

    off_prob = off_pt / denom
    p = 1. - off_prob
    p.name = 'p(%s)' % z_name

    hpart = []
    for i in xrange(r):
        hpart.append([pt_ij / denom for pt_ij in pt[i]])

    h = T.alloc(0., batch_size, ch, zr, zc)

    for i in xrange(r):
        for j in xrange(c):
            h.name = 'h_interm'
            h = T.set_subtensor(h[:, :, i:zr:r, j:zc:c], hpart[i][j])

    h.name = 'h(%s)' % z_name

    if theano_rng is None:
        return p, h
    
    ### --------------------- DONE IF NO SAMPLES ARE GENERATED ---------------------------###
    else:
        events = []
        for i in xrange(r):
            for j in xrange(c):
                events.append(hpart[i][j])
        events.append(off_prob)

        events = [event.dimshuffle(0, 1, 2, 3, 'x') for event in events]

        events = tuple(events)

        stacked_events = T.concatenate(events, axis=4)

        rows = zr // pool_shape[0]
        cols = zc // pool_shape[1]
        outcomes = pool_shape[0] * pool_shape[1] + 1
        assert stacked_events.ndim == 5
        for se, bs, r, c, chv in get_debug_values(stacked_events, batch_size,
                                                  rows, cols, ch):
            assert se.shape[0] == bs
            assert se.shape[1] == r
            assert se.shape[2] == c
            assert se.shape[3] == chv
            assert se.shape[4] == outcomes
        reshaped_events = stacked_events.reshape((
            batch_size * rows * cols * ch, outcomes))

        multinomial = theano_rng.multinomial(pvals=reshaped_events,
                                             dtype=p.dtype)

        reshaped_multinomial = multinomial.reshape((batch_size, ch, rows,
                                                    cols, outcomes))

        h_sample = T.alloc(0., batch_size, ch, zr, zc)

        idx = 0
        for i in xrange(r):
            for j in xrange(c):
                h_sample = T.set_subtensor(h_sample[:, :, i:zr:r, j:zc:c],
                                           reshaped_multinomial[:, :, :, :,
                                           idx])
                idx += 1

        p_sample = 1 - reshaped_multinomial[:, :, :, :, -1]

        return p, h, p_sample, h_sample


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
    def __init__ (self, _motifLength, _numMotifs, _learningRate=0.1, _poolingFactor=1, _alphabet=IUPAC.unambiguous_dna):
        # parameters for the motifs
        self.motifLength = _motifLength
        self.numMotifs = _numMotifs
        self.alphabet = _alphabet
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
        
        
        



# apply the two classes to calculate a forward pass in our algorithm
seqReader = FASTAReader('.')
allSeqs = seqReader.readSequencesFromFile('data/wgEncodeAwgDnaseUwAg10803UniPk.fa')
data = [allSeqs[random.randrange(0,len(allSeqs))] for i in range(20000)]
train_set, test_set = train_test_split(data, test_size=0.1)
print "Training set size: " + str(len(train_set))
print "Test set size: " + str(len(test_set))
start = time.time()
trainingData = np.array([getMatrixFromSeq(t) for t in train_set])
testingData = np.array([getMatrixFromSeq(t) for t in test_set])
print "Conversion of test set in (in ms): " + str((time.time()-start)*1000)
        
        
        
        

        
theano.config.mode='FAST_RUN'
cross_validation_runs = 1
c = ['b', 'g', 'r', 'c', 'm','k','y']
randomColor = lambda: (random.randint(0,1),random.randint(0,1),random.randint(0,1))
count = 1
for i in range(cross_validation_runs):
	learner = CRBM(5, 20, 0.001, 2)
	scores = learner.trainMinibatch(trainingData, testingData, 10, 50, 1)
	plt.ylabel('Average Free Energy')
	plt.xlabel('Epoch')
	plt.title('CD_1, LR: 0.001 , #Motifs: 2, motifLength: 5, batchSize: 50')
	plt.plot(scores, color=c[i])
	
filename = 'training_lrChoice_'+str(count)+'.png'
plt.savefig(filename)
plt.cla()
count += 1
print "Overall training time (in minutes): " + str((time.time()-start)/60.)
