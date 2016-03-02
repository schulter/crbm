import theano.tensor as T
import theano
import numpy as np


class TrainingObserver:
    def __init__(self, _model, _data, _name):
        self.model = _model
        self.data = _data
        self.name = _name
        self.batchSize = 50
        self.scores = []
        self.scoringFunction = self.getScoringFunction()

    def calculateScore(self):
        raise NotImplementedError("Abstract class not callable")

    def getScoringFunction(self):
        raise NotImplementedError("Abstract class not callable")


class FreeEnergyObserver(TrainingObserver):
    def __init__(self, _model, _data, _name="Free Energy Observer"):
        TrainingObserver.__init__(self, _model, _data, _name)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['scoringFunction']
        del state['data']
        del state['model']
        return state

    def calculateScore(self):
        iterations = max(self.data.shape[0] / self.batchSize, 1)
        sumOfScores = 0
        for batchIdx in xrange(iterations):
            sumOfScores += self.scoringFunction(batchIdx)
        score = sumOfScores / iterations  # mean
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
                                     givens={D: dataS[index * self.batchSize:(index + 1) * self.batchSize]},
                                     name='freeEnergyObservation'
                                     )
        return scoringFun

    def getFreeEnergy(self, D):
        return self.model.meanFreeEnergy(D)


class ReconstructionRateObserver(TrainingObserver):
    def __init__(self, _model, _data, _name="Reconstruction Rate Observer"):
        TrainingObserver.__init__(self, _model, _data, _name)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['scoringFunction']
        del state['data']
        del state['model']
        return state

    def calculateScore(self):
        iterations = max(self.data.shape[0] / self.batchSize, 1)
        sumOfScores = 0
        for batchIdx in xrange(iterations):
            sumOfScores += self.scoringFunction(batchIdx)
        count = sumOfScores / iterations  # mean

        # got the mean count of correct letters for all seqs
        self.scores.append(count)
        return count

    def getScoringFunction(self):
        dataS = theano.shared(value=self.data, borrow=True, name='data')

        D = T.tensor4('data')
        index = T.lscalar()
        score = self.getReconstructionRate(D)
        scoringFun = theano.function([index],
                                     score,
                                     allow_input_downcast=True,
                                     givens={D: dataS[index * self.batchSize:(index + 1) * self.batchSize]},
                                     name='ReconstructioinRateObservation'
                                     )
        return scoringFun

    def getReconstructionRate(self, D):
        [_, H] = self.model.computeHgivenV(D)
        [_, V] = self.model.computeVgivenH(H)
        sames = V * D  # elements are 1 if they have the same letter...
        return T.mean(T.sum(sames, axis=2))  # sum over letters, mean over rest


class MotifObserver(TrainingObserver):
    def __init__(self, _model, _data, _name="Motif Error Observer"):
        TrainingObserver.__init__(self, _model, _data, _name)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['scoringFunction']
        del state['data']
        del state['model']
        return state

    def calculateScore(self):
        self.scores.append(self.model.motifs.get_value())

    def getScoringFunction(self):
        return np.sum  # it's a stub


class ParameterObserver(TrainingObserver):
    def __init__(self, _model, _data, _name="Parameter Observer"):
        TrainingObserver.__init__(self, _model, _data, _name)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['scoringFunction']
        del state['data']
        del state['model']
        return state

    def calculateScore(self):
        self.scores.append((self.model.motifs.get_value(), self.model.bias.get_value(), self.model.c.get_value()))

    def getScoringFunction(self):
        return np.sum  # it's a stub


class MotifHitObserver(TrainingObserver):
    def __init__(self, _model, _data, _name="Motif Hit Observer"):
        TrainingObserver.__init__(self, _model, _data, _name)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['scoringFunction']
        del state['data']
        del state['model']
        return state

    def calculateScore(self):
        iterations = max(self.data.shape[0] / self.batchSize, 1)
        sumOfHits = 0
        for batchIdx in xrange(iterations):
            sumOfHits += self.scoringFunction(batchIdx)
        avgHits = sumOfHits / iterations  # mean
        self.scores.append(avgHits)
        return T.mean(avgHits)

    def getScoringFunction(self):
        dataS = theano.shared(value=self.data, borrow=True, name='data')

        D = T.tensor4('data')
        index = T.lscalar()
        hits = self.getMotifHits(D)
        scoringFun = theano.function([index],
                                     hits,
                                     allow_input_downcast=True,
                                     givens={D: dataS[index * self.batchSize:(index + 1) * self.batchSize]},
                                     name='MotifHitInErrorObservation'
                                     )
        return scoringFun

    def getMotifHits(self, D):
        [_, H] = self.model.computeHgivenV(D)
        return T.mean(H, axis=0)  # mean over samples (K x 1 x N_h)


class InformationContentObserver(TrainingObserver):
    def __init__(self, _model, _name="Information Content Observer"):
        TrainingObserver.__init__(self, _model, None, _name)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['scoringFunction']
        del state['model']
        return state

    def calculateScore(self):
        meanIC = self.scoringFunction(self.model.motifs.get_value())
        self.scores.append(meanIC)
        return meanIC

    def getScoringFunction(self):
        M = T.tensor4('Matrix')
        ic = self.getInformationContent(M)
        scoringFun = theano.function([M],
                                     ic,
                                     allow_input_downcast=True,
                                     name='InformationContentObservation'
                                     )
        return scoringFun

    def getInformationContent(self, M):
        pwm = self.model.softmax(M)
        entropy = -pwm * T.log2(pwm)
        entropy = T.sum(entropy, axis=2)  # sum over letters
        return T.log2(self.model.motifs.shape[2]) - T.mean(
            entropy)  # log is possible information due to length of sequence


class MedianICObserver(TrainingObserver):
    def __init__(self, _model, _name="Median Information Content Observer"):
        TrainingObserver.__init__(self, _model, None, _name)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['scoringFunction']
        del state['model']
        return state

    def calculateScore(self):
        meanIC = self.scoringFunction(self.model.motifs.get_value())
        self.scores.append(meanIC)
        return meanIC

    def getScoringFunction(self):
        M = T.tensor4('Matrix')
        medIC = self.getMedianIC(M)
        scoringFun = theano.function([M],
                                     medIC,
                                     allow_input_downcast=True,
                                     name='InformationContentObservation'
                                     )
        return scoringFun

    def getMedianIC(self, M):
        pwm = self.model.softmax(M)
        entropy = -pwm * T.log2(pwm)
        entropy = T.sum(entropy, axis=2)  # sum over letters
        return T.log2(self.model.motifs.shape[2]) - T.mean(T.sort(entropy, axis=2)[:, :, entropy.shape[2] // 2])
