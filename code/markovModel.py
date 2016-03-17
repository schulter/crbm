import numpy as np

class MarkovModel:

    def __init__(self, alphabet=['A', 'C', 'G', 'T']):
        self.TPM = None
        self.alphabet = alphabet
        self.background = np.zeros(len(alphabet))
        self.scores = None

    def trainModel(self, data):
        self.TPM = np.zeros((len(self.alphabet), len(self.alphabet)))
        for sequence in range(data.shape[0]):
            for position in range(data.shape[3]-1):
                from_pos = np.argmax(data[sequence, 0, :, position])
                to_pos = np.argmax(data[sequence, 0, :, position+1])
                self.TPM[from_pos, to_pos] += 1
        
        self.TPM /= (data.shape[0] * (data.shape[3]-1))
        self.background = np.mean(data, axis=(0, 1, 3))
        print self.background
        print self.TPM
        
    def setTPM(self, TPM):
        self.TPM = TPM
    
    def evaluateSequences(self, testset):
        log_probs = np.log(self.TPM)
        print log_probs
        scores = np.zeros(testset.shape[0])
        for sequence in range(testset.shape[0]):
            for position in range(testset.shape[3]-1):
                from_pos = np.argmax(testset[sequence, 0, :, position])
                to_pos = np.argmax(testset[sequence, 0, :, position+1])
                scores[sequence] += log_probs[from_pos, to_pos]
        self.scores = scores
        return scores
