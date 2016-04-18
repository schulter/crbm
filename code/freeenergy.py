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

def getFreeEnergyPoints(model, data):
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

