
import theano.tensor as T
import theano
import numpy as np
import theano.tensor.nnet.conv as conv

class TrainingObserver:
	
	def __init__ (self, _model, _data, _name):
		self.model = _model
		self.data = _data
		self.name = _name
		self.batchSize = 50
		self.scores = []
		self.scoringFunction = self.getScoringFunction()
		
	def calculateScore (self):
		raise NotImplementedError("Abstract class not callable")
		
	def getScoringFunction (self):
		raise NotImplementedError("Abstract class not callable")


class FreeEnergyObserver (TrainingObserver):
	
	def __init__ (self, _model, _data, _name="Free Energy Observer"):
		TrainingObserver.__init__(self, _model, _data, _name)
		
	def __getstate__(self):
		state = dict(self.__dict__)
		del state['scoringFunction']
		del state['data']
		del state['model']
		return state

	def calculateScore (self):
		iterations = max(self.data.shape[0] / self.batchSize, 1)
		sumOfScores = 0
		for batchIdx in xrange(iterations):
			sumOfScores += self.scoringFunction(batchIdx)
		score = sumOfScores / iterations # mean
		self.scores.append(score)
		return score

	def getScoringFunction(self):
		dataS = theano.shared(value=self.data, borrow=True, name='data')

		D = T.tensor4('data')
		index = T.lscalar()
		score = self.getFreeEnergy(D)
		scoringFun = theano.function([index],
					 score,
					 allow_input_downcast=True,
					 givens={D: dataS[index*self.batchSize:(index+1)*self.batchSize]},
					 name='freeEnergyObservation'
		)
		return scoringFun


	def getFreeEnergy (self, D):
		# firstly, compute hidden part of free energy
		C = conv.conv2d(D, self.model.motifs)
		bMod = self.model.bias # to prevent member from being shuffled
		bMod = bMod.dimshuffle('x', 1, 0, 'x') # add dims to the bias on both sides
		
		C = C + bMod
		hiddenPart = T.sum(T.log(1. + T.exp(C)), axis=1) # dim: N_batch x 1 x N_h after sum over K
		hiddenPart = T.sum(T.mean(hiddenPart, axis=0)) # mean over all samples and sum over units
		
		# compute the visible part
		cMod = self.model.c
		cMod = cMod.dimshuffle('x', 0, 1, 'x') # make it 4D and broadcastable there
		visiblePart = T.mean(D * cMod, axis=0) # dim: 1 x 4 x N_v
		visiblePart = T.sum(visiblePart)
		
		free_energy = -hiddenPart - visiblePart # don't return the negative because it's more difficult to plot
		
		return free_energy


class ReconstructionErrorObserver (TrainingObserver):
	
	def __init__(self, _model, _data, _name="Reconstruction Error Observer"):
		TrainingObserver.__init__(self, _model, _data, _name)
		
	def __getstate__(self):
		state = dict(self.__dict__)
		del state['scoringFunction']
		del state['data']
		del state['model']
		return state

	def calculateScore (self):
		iterations = max(self.data.shape[0] / self.batchSize, 1)
		sumOfScores = 0
		for batchIdx in xrange(iterations):
			sumOfScores += self.scoringFunction(batchIdx)
		count = sumOfScores / iterations # mean
		
		# got the count of correct letters for all seqs
		# now, make it percentage of sequence (prob. to get all correct)
		count /= self.data.shape[3]
		error = 1 - count # how much error do we have
		
		self.scores.append(error)
		return error
		
	
	def getScoringFunction(self):
		dataS = theano.shared(value=self.data, borrow=True, name='data')

		D = T.tensor4('data')
		index = T.lscalar()
		score = self.getReconstructionError(D)
		scoringFun = theano.function([index],
					 score,
					 allow_input_downcast=True,
					 givens={D: dataS[index*self.batchSize:(index+1)*self.batchSize]},
					 name='ReconstructioinErrorObservation'
		)
		return scoringFun


	def getReconstructionError (self, D):
		[H, S_H] = self.model.forwardBatch(D)
		V = self.model.backwardBatch(S_H)
		S_V = self.model.sampleVisibleLayer(V)
		sames = S_V * D # elements are 1 if they have the same letter...
		count = T.sum(T.mean(sames, axis=0)) # mean over samples, sum over rest
		return count


class MotifObserver (TrainingObserver):
	
	def __init__(self, _model, _data, _name="Motif Error Observer"):
		TrainingObserver.__init__(self, _model, _data, _name)
	
	def __getstate__(self):
		state = dict(self.__dict__)
		del state['scoringFunction']
		del state['data']
		del state['model']
		return state

	def calculateScore (self):
		self.scores.append(self.model.motifs.get_value())
		
	def getScoringFunction (self):
		return np.sum # it's a stub



class MotifHitObserver (TrainingObserver):

	def __init__(self, _model, _data, _name="Motif Hit Observer"):
		TrainingObserver.__init__(self, _model, _data, _name)

	def __getstate__(self):
		state = dict(self.__dict__)
		del state['scoringFunction']
		del state['data']
		del state['model']
		return state


	def calculateScore (self):
		iterations = max(self.data.shape[0] / self.batchSize, 1)
		sumOfHits = 0
		for batchIdx in xrange(iterations):
			sumOfHits += self.scoringFunction(batchIdx)
		avgHits = sumOfHits / iterations # mean
		
		
		self.scores.append(avgHits)
		return avgHits.mean()


	def getScoringFunction(self):
		dataS = theano.shared(value=self.data, borrow=True, name='data')

		D = T.tensor4('data')
		index = T.lscalar()
		hits = self.getMotifHits(D)
		scoringFun = theano.function([index],
					 hits,
					 allow_input_downcast=True,
					 givens={D: dataS[index*self.batchSize:(index+1)*self.batchSize]},
					 name='MotifHitInErrorObservation'
		)
		return scoringFun


	def getMotifHits (self, D):
		[H, S_H] = self.model.forwardBatch(D)
		return T.mean(H, axis=0) # mean over samples (K x 1 x N_h)
