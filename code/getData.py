import Bio.SeqIO as sio
import Bio.motifs.matrix as mat
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio import motifs

import numpy as np
import re



L=dict({'A':0,'a':0,'C':1,'c':1,'G':2,'g':2,'T':3,'t':3})

def getOneHotSeq(seq):
    m = len(seq.alphabet.letters)
    n = len(seq)
    # why can't it just be np.zeros((m,n),dtype=np.float32)
    result = np.zeros((1, m, n), dtype=np.float32)
    for i in range(len(seq)):
        result[0,L[seq[i]],i]= 1
    return result

"""
This class reads sequences from fasta files.
To use it, create an instance of that object and use
the function readSequencesFromFile.
"""
class SeqReader:
    def readSequencesFromFile (self, filename):
        dhsSequences = []
        for dhs in sio.parse(open(filename), 'fasta', IUPAC.unambiguous_dna):
            match=re.search(r'N',str(dhs.seq), re.I)
            if match:
                print "skip sequence containing N"
		continue
            dhsSequences.append(getOneHotSeq(dhs.seq))
        return dhsSequences
    
    
    

class JASPARReader:
    def readSequencesFromFile (self, filename):
        matrices = []
        for mat in motifs.parse(open(filename), 'jaspar'):
            matrices.append(mat.pwm)
        return matrices

