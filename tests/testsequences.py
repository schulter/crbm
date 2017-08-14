import os
import numpy as np
from crbm import readSeqsFromFasta, seqToOneHot
from crbm import splitTrainingTest

def test_trainingtestsplit():

    path = os.path.join(os.path.dirname(__file__), '..', 'crbm', 'data')
    fafile = os.path.join(path, 'oct4.fa')
    seqs = readSeqsFromFasta(fafile)
    onehot = seqToOneHot(seqs)
    
    for ndata in (None, 10):
        # Without randomization
        splitTrainingTest(fafile, 0.1, num_top_regions = ndata, randomize = False)
        
        fafile = os.path.join(path, 'oct4_train.fa')
        onehot_train = seqToOneHot(readSeqsFromFasta(fafile))

        fafile = os.path.join(path, 'oct4_test.fa')
        onehot_test = seqToOneHot(readSeqsFromFasta(fafile))

        conehot = np.concatenate((onehot_test, onehot_train), axis=0)

        if type(ndata) == type(None):
            np.testing.assert_equal(conehot,  onehot)
        else:
            np.testing.assert_equal(conehot,  onehot[:10])

        # With randomization
        splitTrainingTest(fafile, 0.1, randomize = True)
        
        fafile = os.path.join(path, 'oct4_train.fa')
        onehot_train = seqToOneHot(readSeqsFromFasta(fafile))

        fafile = os.path.join(path, 'oct4_test.fa')
        onehot_test = seqToOneHot(readSeqsFromFasta(fafile))

        conehot = onehot_train.shape[0] + onehot_test.shape[0]
        np.testing.assert_equals(conehot,  onehot.shape[0] if type(ndata) == type(None) else ndata)

    os.path.unlink(os.path.join(path, 'oct4_train.fa'))
    os.path.unlink(os.path.join(path, 'oct4_test.fa'))
