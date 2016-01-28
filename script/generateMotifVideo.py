# some always important inputs
import sys
import os
import random
import time
import numpy as np
import cPickle
import math

# the underlying convRBM implementation
sys.path.append(os.path.abspath('../code'))
from convRBM import CRBM
import getData as dataRead

# plotting and data handling
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

# the biopython stuff
import Bio.SeqIO as sio
import Bio.motifs.matrix as mat
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio import motifs as mot

######## HYPER PARAMS ###########
#################################
date_string = '2016_01_27_21_27'
numberOfEpochs = 5
#################################

# read in the model
modelDir = '../../training/' + date_string
learner = CRBM(file_name = modelDir + '/model.pkl')

def getLetterToInt (num):
    if num == 0:
        return 'A'
    elif num == 1:
        return 'C'
    elif num == 2:
        return 'G'
    elif num == 3:
        return 'T'
    else:
        print 'ERROR: Num ' + str(num) + " not a valid char in DNA alphabet"
        return -1

def createMotifFromMatrix (matrix, alphabet=IUPAC.unambiguous_dna):
    assert matrix.shape[0] == 4
    
    # transform the matrix such that the log odds are taken away
    # matrix_ij = log(foreground/background) <=> log(foreground) - log(background)
    psm = matrix + np.log(0.25) # 0.25 if we treat all letters as equally probable
    psm = np.exp(psm)
    psm = psm / psm.sum(axis=1, keepdims=True)
    
    # make this matrix a valid motif
    counts = {}
    for row in range(4):
        counts[getLetterToInt(row)] = (psm[row]).tolist()
    motif = mot.Motif(alphabet=alphabet, instances=None, counts=counts)
    return motif


def weblogo(motif, file_format="png", version="2.8.2", **kwds): 
    from Bio._py3k import urlopen, urlencode, Request 
    frequencies = motif.format('transfac') 
    url = 'http://weblogo.threeplusone.com/create.cgi' 
    values = {'sequences': frequencies, 
                    'format': file_format.lower(), 
                    'stack_width': 'medium', 
                    'stack_per_line': '40', 
                    'alphabet': 'alphabet_dna', 
                    'ignore_lower_case': True, 
                    'unit_name': "bits", 
                    'first_index': '1', 
                    'logo_start': '1', 
                    'logo_end': str(motif.length), 
                    'composition': "comp_auto", 
                    'percentCG': '', 
                    'scale_width': True, 
                    'show_errorbars': False, 
                    'logo_title': '', 
                    'logo_label': '', 
                    'show_xaxis': False, 
                    'xaxis_label': '', 
                    'show_yaxis': False, 
                    'yaxis_label': '', 
                    'yaxis_scale': 'auto', 
                    'yaxis_tic_interval': '1.0', 
                    'show_ends': False, 
                    'show_fineprint': False, 
                    'color_scheme': 'color_auto', 
                    'symbols0': '', 
                    'symbols1': '', 
                    'symbols2': '', 
                    'symbols3': '', 
                    'symbols4': '', 
                    'color0': '', 
                    'color1': '', 
                    'color2': '', 
                    'color3': '', 
                    'color4': '', 
                    } 
    values.update(dict((k, "" if v is False else str(v)) for k, v in kwds.items()))
    data = urlencode(values).encode("utf-8")
    req = Request(url, data)
    response = urlopen(req)
    return response


def getLogoListFrom4DMatrix(matrix):
    images = []
    for motifNum in range(matrix.shape[0]):
        m = createMotifFromMatrix(matrix[motifNum,0])
        reader = weblogo(m)
        images.append(plt.imread(reader))
    return images


bestSplit = lambda x: (round(math.sqrt(x)), math.ceil(x / round(math.sqrt(x))))

def getObserverIndex():
    count = 0
    for obs in learner.observers:
        if "motif" in obs.name.lower():
            return count
        count += 1

learner.printHyperParams()
print len(learner.observers[getObserverIndex()].scores)

# get the logos for all scores during training
observerIndex = getObserverIndex()
logosOverTime = []
frames = min(len(learner.observers[observerIndex].scores), numberOfEpochs)
for timeSlice in range(frames):
    allMotifsPerSlice = learner.observers[observerIndex].scores[timeSlice]
    logosOverTime.append(getLogoListFrom4DMatrix(allMotifsPerSlice))
    print "Got Logos for Time/Epoch " + str(timeSlice)


from matplotlib import animation

fig = plt.figure(figsize=(30,13))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
frame_text = fig.text(0.95, 0.95,
                      'Epoch: ' + str(0),
                      verticalalignment='bottom',
                      horizontalalignment='right',
                      color='green', fontsize=30)
allMotifsOverTime = learner.observers[observerIndex].scores
print allMotifsOverTime[0].shape

x, y = bestSplit(allMotifsOverTime[0].shape[0])
print x, y
axesList = []

def init():
    print "in init"
    for i in range(allMotifsOverTime[0].shape[0]):
        ax = fig.add_subplot(x, y, i+1, xticks=[], yticks=[])
        im = ax.imshow(logosOverTime[0][i])
        axesList.append(im)
    print len(axesList)
        
def printFrame(frameNr):
    numMotifs = allMotifsOverTime[frameNr].shape[0]
    for motif in range(numMotifs):
        axesList[motif].set_data(logosOverTime[frameNr][motif])
        #ax.imshow(logosOverTime[frameNr][motif])
    frame_text.set_text('Epoch: ' + str(frameNr))

anim = animation.FuncAnimation(fig,
                               printFrame,
                               init_func=init,
                               frames=frames,
                               interval=200, repeat=True)

anim.save(modelDir + '/motifChanges.mp4', fps=10)
