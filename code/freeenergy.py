import numpy as np
import theano
from theano import tensor as T

def getFreeEnergyFunction (model, data):
    D = T.tensor4('data')
    batchSize=model.hyper_params['batch_size']
    dataS = theano.shared(value=data, borrow=True, name='givenData')
    index = T.lscalar()
    energy = model.freeEnergyForData(D)
    return theano.function([index], energy, allow_input_downcast=True,
                           givens={D: dataS[index*batchSize:(index+1)*batchSize]},
                           name='freeDataEnergy'
                          )

def getMeanFreeEnergyFunction (model, data):
    D = T.tensor4('data')
    batchSize=model.hyper_params['batch_size']
    dataS = theano.shared(value=data, borrow=True, name='givenData')
    index = T.lscalar()
    energy = model.meanFreeEnergy(D)
    return theano.function([index], energy, allow_input_downcast=True,
                           givens={D: dataS[index*batchSize:(index+1)*batchSize]},
                           name='freeDataEnergy'
                          )

def getFreeEnergyPoints(model, data):
    nseq=int((data.shape[3]-model.hyper_params['motif_length'] + 1)/model.hyper_params['pooling_factor'])*\
        		model.hyper_params['pooling_factor']+ model.hyper_params['motif_length'] -1
    data=data[:,:,:,:nseq]
    fun = getFreeEnergyFunction(model, data)
    batchSize=model.hyper_params['batch_size']
    iterations = data.shape[0] // batchSize

    M = np.zeros(data.shape[0])
    for batchIdx in xrange(iterations):
        #print "Setting from idx " + str(batchIdx*batchSize) + " to " + str((batchIdx+1)*batchSize)
        M[batchIdx*batchSize:(batchIdx+1)*batchSize] = fun(batchIdx)
    
    # to clean up the rest
    if data.shape[0] > iterations*batchSize:
        M[(batchIdx+1)*batchSize:] = fun(batchIdx+1)
    return M

def getMeanFreeEnergy(model, data):
    nseq=int((data.shape[3]-model.hyper_params['motif_length'] + 1)/model.hyper_params['pooling_factor'])*\
        		model.hyper_params['pooling_factor']+ model.hyper_params['motif_length'] -1
    data=data[:,:,:,:nseq]
    fun = getMeanFreeEnergyFunction(model, data)
    batchSize=model.hyper_params['batch_size']
    iterations = data.shape[0] // batchSize

    #M = np.zeros(data.shape[0])
    m=0
    for batchIdx in xrange(iterations):
        #print "Setting from idx " + str(batchIdx*batchSize) + " to " + str((batchIdx+1)*batchSize)
        m=m+ fun(batchIdx)
    
    # to clean up the rest
    if data.shape[0] > iterations*batchSize:
        m = m+ fun(batchIdx+1)
        iterations+1
    m=m/iterations
    return m

