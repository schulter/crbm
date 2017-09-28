import os
import shutil
import numpy as np
from secomo import readSeqsFromFasta, seqToOneHot
from secomo import splitTrainingTest, load_sample

def test_trainingtestsplit(tmpdir):

    path = tmpdir.mkdir("data")
    shutil.copyfile(os.path.join(os.path.dirname(__file__), \
            '..', 'secomo', 'data', 'oct4.fa'),
            os.path.join(path.strpath, 'oct4.fa'))

    fafile = os.path.join(path.strpath, 'oct4.fa')
    onehot = load_sample()

    for ndata in [None, 10]:
        # Without randomization
        splitTrainingTest(fafile, 0.1,
                num_top_regions = ndata, randomize = False)

        onehot_train = seqToOneHot(readSeqsFromFasta(
            os.path.join(path.strpath, 'oct4_train.fa')))

        onehot_test = seqToOneHot(readSeqsFromFasta(
            os.path.join(path.strpath, 'oct4_test.fa')))

        conehot = np.concatenate((onehot_test, onehot_train), axis=0)

        if type(ndata) == type(None):
            np.testing.assert_equal(conehot,  onehot)
        else:
            np.testing.assert_equal(conehot,  onehot[:10])

        # With randomization
        splitTrainingTest(fafile, 0.1, num_top_regions = ndata,
                randomize = True)

        onehot_train = seqToOneHot(readSeqsFromFasta(
            os.path.join(path.strpath, 'oct4_train.fa')))

        onehot_test = seqToOneHot(readSeqsFromFasta(
            os.path.join(path.strpath, 'oct4_test.fa')))

        conehot = onehot_train.shape[0] + onehot_test.shape[0]
        np.testing.assert_equal(conehot,
                onehot.shape[0] if type(ndata) == type(None) else ndata)
