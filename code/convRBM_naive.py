# Imports
import sys
import os
import random
import time
import numpy as np


class NaiveCRBM:

    def __init__ (self, hyperParams):
        self.hyper_params = hyperParams
        self.motifs = np.zeros((2 * self.hyper_params['number_of_motifs'], 1, 4, self.hyper_params['motif_length']))
        self.bias = np.zeros((1, self.hyper_params['number_of_motifs']))
        self.c = np.zeros((1,4))
        
    def setMotifs (self, motifs_):
        self.motifs = motifs_

    def setBiases (self, b_):
        self.bias = b_

    def setC (self, c_):
			  self.c=c_

    def computeHgivenV (self, data):
        N_h = data.shape[3]-self.hyper_params['motif_length']+1

        M=self.hyper_params['motif_length']

        if self.hyper_params['doublestranded']:
            K=2*self.hyper_params['number_of_motifs']
        else:
            K=self.hyper_params['number_of_motifs']

        h_input = np.zeros((data.shape[0], K, 1, N_h))
        for sample in range(data.shape[0]):
            for k in range(K):
                for n in range(N_h):
                    x = np.sum(data[sample,0,:,range(n,n+M)].T*self.motifs[k,0,:,:]) + self.bias[0,k]
                    h_input[sample, k, 0, n] = x
        
        prob_of_H = np.exp(h_input)
        H = np.zeros(h_input.shape)

        horizontal_pooling_factor=self.hyper_params['pooling_factor']
        if self.hyper_params['doublestranded']:
            vertical_pooling_factor=2
        else:
            vertical_pooling_factor=1

        horizontal_bins = N_h / self.hyper_params['pooling_factor']
        vertical_bins = self.hyper_params['number_of_motifs']

        for iseq in range(data.shape[0]):
            for ivbin in range(vertical_bins):
                for ihbin in range(horizontal_bins):
                    
                    # first, get the denominator
                    denominator = 1.0
                    for iv in range(vertical_pooling_factor):
                        for ih in range(horizontal_pooling_factor):
                	      	 denominator+=prob_of_H[iseq,ivbin*vertical_pooling_factor+iv,0,ihbin*horizontal_pooling_factor+ih]
                    
                    # now we can do the softmax in there
                    arr_of_probs = []
                    for iv in range(vertical_pooling_factor):
                        for ih in range(horizontal_pooling_factor):
                            k_pos = ivbin*vertical_pooling_factor+iv
                            pool_pos = ihbin*horizontal_pooling_factor+ih
                            softmax_val = prob_of_H[iseq,k_pos,0,pool_pos]/denominator
                            prob_of_H[iseq,k_pos,0,pool_pos] = softmax_val
                            arr_of_probs.append(softmax_val)
                    
                    # and at last, do the sampling
                    arr_of_probs.append(1./denominator) #last option means no one is on
                    sample = np.random.multinomial(n=1, pvals=np.array(arr_of_probs))
                    pos = np.argmax(sample)
                    if pos < len(sample)-1: # otherwise, the No-Unit-On option was chosen
                        k_pos = ivbin*vertical_pooling_factor + (pos % vertical_pooling_factor)
                        pool_pos = ihbin*horizontal_pooling_factor + (pos // horizontal_pooling_factor)
                        H[iseq,k_pos,0,pool_pos] = 1 # set sample
                        
        return [prob_of_H,H]


    def computeVgivenH (self, hidden):
        
        # calculate full convolution (not valid, therefore padding is applied with zeros)
        M = self.hyper_params['motif_length']
        N_v = hidden.shape[3] + M - 1
        if self.hyper_params['doublestranded']:
            K=2*self.hyper_params['number_of_motifs']
        else:
            K=self.hyper_params['number_of_motifs']

        v_input = np.zeros((hidden.shape[0],1,4,N_v))
        for i in range(4):
            v_input[:,:,i,:]=self.c[0,i]

        for iseq in range(hidden.shape[0]):
            for k in range(K):
                for n in range(hidden.shape[3]):
                    v_input[iseq,0,:,range(n,n+M)] += \
                    		self.motifs[k,0,:,:].T * hidden[iseq,k,0,n]
                        
        
        prob_of_V = self.softmax(v_input)
        
        V = np.zeros(v_input.shape)
        for sample in range(prob_of_V.shape[0]):
            for col in range(prob_of_V.shape[3]):
                V[sample,0,:,col] = np.random.multinomial(n=1,pvals=prob_of_V[sample,0,:,col],size=1)

        return [prob_of_V, V]
        

    def collectVHStatistics(self, prob_of_H, data):
    	  #reshape input 
        # calculate full convolution (not valid, therefore padding is applied with zeros)
        M = self.hyper_params['motif_length']
        if self.hyper_params['doublestranded']:
            K=2*self.hyper_params['number_of_motifs']
        else:
            K=self.hyper_params['number_of_motifs']

        vh = np.zeros((K,1,4,M))

        for iseq in range(prob_of_H.shape[0]):
            for k in range(K):
                for n in range(prob_of_H.shape[3]):
                    vh[k,0,:,:] += data[iseq,0,:,range(n,n+M)].T * prob_of_H[iseq,k,0,n]
                        
        
        vh=vh/ (prob_of_H.shape[0]*prob_of_H.shape[3])

        return vh

    def collectVStatistics(self, data):
        #reshape input
        c=np.zeros((1,4))
        for iseq in range(data.shape[0]):
            for ipos in range(data.shape[3]):
                c[0,:] += data[iseq,0,:,ipos] + data[iseq,0,::-1,ipos]
        c=2.*c/np.sum(c)
        return c

    def collectHStatistics(self, hidden):
        #reshape input 
        K=self.hyper_params['number_of_motifs']
        if self.hyper_params['doublestranded']:
            K=2*K
        b=np.zeros((1,K))

        for iseq in range(hidden.shape[0]):
            for ipos in range(hidden.shape[3]):
                for k in range(K):
                    b[0,k]+=hidden[iseq,k,0,ipos]

        b=b/(hidden.shape[0]*hidden.shape[3])
        return b

    def collectUpdateStatistics(self, prob_of_H, data):
        #reshape input 

        average_VH=self.collectVHStatistics(prob_of_H, data)
        average_H=self.collectHStatistics(prob_of_H)
        average_V=self.collectVStatistics(data)

        # make the kernels respect the strand structure
        if self.hyper_params['doublestranded']:
            average_VH,average_H = self.matchWeightchangeForComplementaryMotifs(average_VH,average_H)

        return average_VH, average_H, average_V

    
    def matchWeightchangeForComplementaryMotifs(self, evh,eh):

        evhre = evh.reshape((evh.shape[0]//2, 2, 1,evh.shape[2], evh.shape[3]))
        evhre_ = T.inc_subtensor(evhre[:,0,:,:,:], evhre[:,1,:,::-1,::-1])
        evhre = T.set_subtensor(evhre[:,1,:,:,:], evhre[:,0,:,::-1,::-1])
        evh=evhre.reshape(evh.shape)
        evh=evh/2.


        ehre = eh.reshape((1,eh.shape[1]//2, 2))
        ehre=T.inc_subtensor(ehre[:,:,0], ehre[:,:,1])
        ehre=T.set_subtensor(ehre[:,:,1], ehre[:,:,0])
        eh=ehre.reshape(eh.shape)
        eh=eh/2.

        return evh,eh


    def updateWeightsOnMinibatch (self, D, numOfCDs):
        # calculate the data gradient for weights (motifs) and bias
        [P_H_data, H_data] = self.computeHgivenV(D)
        if self.hyper_params['verbose']:
            print "Hidden Layer Probabilities:"
            print P_H_data
            print "Hidden Layer Sample:"
            print H_data

        # calculate data gradients
        [G_motif_data, G_bias_data, G_c_data] = self.collectUpdateStatistics(P_H_data, D)

        if self.hyper_params['verbose']:
            print "Data gradient for motifs"
            print G_motif_data

        # calculate model probs
        H = H_data
        for i in range(numOfCDs):
            [P_V, V] = self.computeVgivenH(H)
            if self.hyper_params['verbose']:
                print "Visible Sample for CD " + str(i)
                print V
            [P_H_model, H] = self.computeHgivenV(V)

        # compute the model gradients
        [G_motif_model, G_bias_model, G_c_model] = self.collectUpdateStatistics(P_H_model, V)

        if self.hyper_params['verbose']:
            print "Model gradient for motifs:"
            print G_motif_model
        
        # update the parameters
        grad_motifs = self.hyper_params['learning_rate'] * (G_motif_data - G_motif_model)
        grad_bias = self.hyper_params['learning_rate'] * (G_bias_data - G_bias_model)
        grad_c = self.hyper_params['learning_rate'] * (G_c_data - G_c_model)

        self.motifs += grad_motifs
        self.bias += grad_bias
        self.c += grad_c


    def trainModel (self, trainData):
        batchSize = self.hyper_params['batch_size']
        iterations = trainData.shape[0] / batchSize
        for epoch in range(self.hyper_params['epochs']):
            #print "Params at beginning of epoch [" + str(epoch) + "]"
            #print self.motifs
            #print self.bias
            #print self.c
            for batchIdx in range(iterations):
                self.updateWeightsOnMinibatch(trainData[batchIdx*batchSize:(batchIdx+1)*batchSize], self.hyper_params['cd_k'])
            
            print "[Epoch " + str(epoch) + "] done!"

    def softmax (self, x):
        return np.exp(x) / np.exp(x).sum(axis=2, keepdims=True)
