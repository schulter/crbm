# Imports
import sys
import os
import random
import time
import numpy as np


class NaiveCRBM:

    def __init__ (self, motifLength=1, numMotifs=1, learningRate=0.1, poolingFactor=1):
        self.numberOfKernels = numMotifs
        self.kernelLength = motifLength
        self.poolingFactor = poolingFactor
        self.learningRate = learningRate
        self.setParamsToZero = True
        self.debug = True
        self.updateWeights = True
        if self.setParamsToZero:
            self.kernels = np.zeros((self.numberOfKernels, 1, 4, self.kernelLength))
            self.bias = np.zeros(self.numberOfKernels)
            self.c = np.zeros(4)
        else:
            self.kernels = np.random.rand(self.numberOfKernels, 1, 4, self.kernelLength)
            self.bias = np.random.rand(self.numberOfKernels)
            self.c = np.random.rand(4)
    
    def setCustomKernels (self, kernels):
        self.numberOfKernels = kernels.shape[0]
        self.kernelLength = kernels.shape[3]
        self.kernels = kernels.astype(float)
        if self.setParamsToZero:
            self.bias = np.zeros(self.numberOfKernels)
        else:
            self.bias = np.random.rand(self.numberOfKernels)

    def initializeMotifs (self):
        pass
        
    def complement (self, kernelSlice):
        return kernelSlice[::-1]

    def computeHgivenV (self, data):
        N_h = data.shape[3]-self.kernelLength+1
        H = np.zeros((data.shape[0], self.numberOfKernels, 1, N_h))
        for sample in range(data.shape[0]):
            for k in range(self.numberOfKernels):
                for n in range(N_h):
                    for m in range(self.kernelLength):
                        # calculate the x_i, that is the cross-correlation
                        x = data[sample,0,:,n+m].T.dot(self.kernels[k,0,:,m]) + self.bias[k]
                        #cKernel = self.complement(self.kernels[k,0,:,self.kernelLength-m-1])
                        #x_prime = data[sample,0,:,n+m].T.dot(cKernel) + self.bias[k]
                        H[sample, k, 0, n] += x # + x_prime
        
        if self.debug:
            print "Pre Sigmoid Hidden Layer:"
            print H
        # perform prob max pooling
        P = np.zeros(H.shape)
        S = np.zeros(H.shape)
        H_exp = np.exp(H)
        numBins = N_h / self.poolingFactor
        for sample in range(data.shape[0]):
            for k_pos in range(0, self.numberOfKernels, 1):
                for unit in range(numBins):
                    #print "Doing unit: " + str(unit)
                    # calculate sum within unit
                    sumInUnit = 0
                    for cell in range(self.poolingFactor):
                        curPos = unit*self.poolingFactor+cell
                        sumInUnit += H_exp[sample,k_pos,0,curPos]# + H_exp[sample,k_pos+1,0,curPos]
                        
                    # now, calculate the single positions in P
                    arr = []
                    for cell in range(self.poolingFactor):
                        curPos = unit*self.poolingFactor+cell
                        P[sample,k_pos,0,curPos] = H_exp[sample,k_pos,0,curPos] / (sumInUnit + 1)
                        #P[sample,k_pos+1,0,curPos] = H_exp[sample,k_pos+1,0,curPos] / (sumInUnit + 1)
                        arr.append(P[sample,k_pos,0,curPos])
                        #arr.append(P[sample,k_pos+1,0,curPos])
                    
                    # finally, do the sampling step
                    arr.append(1 / (sumInUnit+1))
                    s = np.random.multinomial(n=1, pvals=np.array(arr),size=1)
                    am = np.argmax(s)
                    if am < self.poolingFactor:#*2:
                        strand = am % 2
                        pos = unit * self.poolingFactor + am #(am // 2)
                        #print "Strand: " + str(strand) + " Pos: " + str(pos)
                        S[sample,k_pos,0,pos] = 1
        return [P,S]


    def computeVgivenH (self, H):
        
        # calculate full convolution (not valid, therefore padding is applied with zeros)
        N_v = H.shape[3] + self.kernelLength - 1
        pad = self.kernelLength-1
        V = np.zeros((H.shape[0],1,4,N_v))
        Y = np.zeros(V.shape)
        H_pad = np.pad(H,[(0,0),(0,0),(0,0),(pad, pad)], 'constant',constant_values=(0,0))
        for sample in range(H.shape[0]):
            for k in range(self.numberOfKernels):
                for n in range(N_v):
                    for m in range(self.kernelLength):
                        Y[sample,0,:,n] += self.kernels[k,0,:,m] * H_pad[sample,k,0,pad+n-m]
                        
        # calculate softmax on convolved data
        P_V = self.softmax(Y)
        
        # sample the visible layer from probabilities
        V = np.zeros(P_V.shape)
        for sample in range(P_V.shape[0]):
            for col in range(P_V.shape[3]):
                V[sample,0,:,col] = np.random.multinomial(n=1,pvals=P_V[sample,0,:,col],size=1)
        
        return [P_V, V]
        

    def collectUpdateStatistics (self, H, data):
        G = np.zeros(self.kernels.shape)
        for sample in range(data.shape[0]):
            for k in range(self.numberOfKernels):
                for n_h in range(H.shape[3]):
                    for m in range(self.kernelLength):
                        G[k,0,:,m] += data[sample,0,:,n_h+m] * H[sample,k,0,n_h]

        der_bias = np.mean(np.mean(H, axis=3), axis=0).reshape(-1)
        der_c = np.mean(np.mean(data, axis=3), axis=0).reshape(-1)
        return [G, der_bias, der_c]
    
    def updateWeightsOnMinibatch (self, D, numOfCDs):
        # calculate the data gradient for weights (motifs) and bias
        [P_H_data, H_data] = self.computeHgivenV(D)
        if self.debug:
            print "Hidden Layer Probabilities:"
            print P_H_data
            print "Hidden Layer Sample:"
            print H_data

        # calculate data gradients
        [G_motif_data, G_bias_data, G_c_data] = self.collectUpdateStatistics(P_H_data, D)

        if self.debug:
            print "Data gradient for motifs"
            print G_motif_data

        # calculate model probs
        H = H_data
        for i in range(numOfCDs):
            [P_V, V] = self.computeVgivenH(H)
            if self.debug:
                print "Visible Sample for CD " + str(i)
                print V
            [P_H_model, H] = self.computeHgivenV(V)
        
        # compute the model gradients
        [G_motif_model, G_bias_model, G_c_model] = self.collectUpdateStatistics(P_H_model, V)
        
        if self.debug:
            print "Model gradient for motifs:"
            print G_motif_model
        
        # update the parameters
        new_kernels = self.learningRate * (G_motif_data - G_motif_model)
        new_bias = self.learningRate * (G_bias_data - G_bias_model)
        new_c = self.learningRate * (G_c_data - G_c_model)

        if self.updateWeights:
            self.kernels += new_kernels
            self.bias += new_bias
            self.c += new_c

        return (new_kernels, new_bias, new_c)

        
    def trainModel (self, trainData, epochs, batchSize, numOfCDs):
        iterations = trainData.shape[0] / batchSize
        for epoch in range(epochs):
            for batchIdx in range(iterations):
                self.updateWeightsOnMinibatch(trainData[batchIdx*batchSize:(batchIdx+1)*batchSize], numOfCDs)
        
    def softmax (self, x):
        return np.exp(x) / np.exp(x).sum(axis=2, keepdims=True)
