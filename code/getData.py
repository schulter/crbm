import Bio.SeqIO as sio
from Bio.Alphabet import IUPAC
from Bio import motifs

import cPickle
import gzip

import numpy as np
import re
import os

L = {'A': 0,
     'a': 0,
     'C': 1,
     'c': 1,
     'G': 2,
     'g': 2,
     'T': 3,
     't': 3
     }


def getOneHotSeq(seq):
    m = len(seq.alphabet.letters)
    n = len(seq)
    result = np.zeros((1, m, n), dtype=np.float32)
    for i in range(len(seq)):
        result[0, L[seq[i]], i] = 1
    return result

"""
This class reads sequences from fasta files.
To use it, create an instance of that object and use
the function readSequencesFromFile.
"""


class SeqReader:

    def __init__(self):
        pass

    def readSequencesFromFile(self, filename):
        dhsSequences = []
        for dhs in sio.parse(open(filename), 'fasta', IUPAC.unambiguous_dna):
            match = re.search(r'N', str(dhs.seq), re.I)
            if match:
                print "skip sequence containing N"
                continue
            dhsSequences.append(getOneHotSeq(dhs.seq))
        return dhsSequences

    def readOneHotFromFile(self, filename):
        f = gzip.open(filename, "rb")
        dhsSequences = cPickle.load(f)
        f.close()
        return dhsSequences

    def writeOneHotToFile(self, filename, seq):
        f = gzip.open(filename, "wb")
        cPickle.dump(seq, f, -1)
        f.close()
        
    def readSeqsInDirectory(self, dirName, matformat="oneHot"):
        listOfSeqs = []
        for filename in os.listdir(dirName):
            if filename.endswith('.fa'):
                path = os.path.join(dirName, filename)
                if matformat == "oneHot":
                    listOfSeqs += self.readSequencesFromFile(path)
        return listOfSeqs
    

class JASPARReader:

    def __init__(self):
        pass

    def readSequencesFromFile(self, filename):
        matrices = []
        for mat in motifs.parse(open(filename), 'jaspar'):
            matrices.append(mat.pwm)
        return matrices


def computeKmerCounts(data, k):
  nseq=data.shape[0]
  seqlen=data.shape[3]
  countmatrix=np.zeros((nseq,np.power(4,k))).astype('int')
  x=np.power(4,range(k))
  a=range(4)
  for i in range(nseq):
    for j in range(seqlen-k+1):
      countmatrix[i,np.dot(np.dot(a,data[i,0,:,j:(j+k)]),x).astype('int')]=\
          countmatrix[i,np.dot(np.dot(a,data[i,0,:,j:(j+k)]),x).astype('int')]+1
  return countmatrix
