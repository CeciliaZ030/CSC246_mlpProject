import numpy as np
import pickle
import random


''' An implementation of an MLP with a single layer of hidden units. '''
class MLP:
    __slots__ = ('W1', 'b1', 'a1', 'z1', 'W2', 'b2', 'a2', 'din', 'dout', 'hidden_units')

    def __init__(self, din, dout, hidden_units):
        ''' Initialize a new MLP with tanh activation and softmax outputs.
        
        Params:
        din -- the dimension of the input data
        dout -- the dimension of the desired output
        hidden_units -- the number of hidden units to use

        Weights are initialized to uniform random numbers in range [-1,1).
        Biases are initalized to zero.
        
        Note: a1 and z1 can be used for caching during backprop/evaluation.
              a2 added.
        
        '''
        self.din = din
        self.dout = dout
        self.hidden_units = hidden_units

        self.a1 = None
        self.z1 = None 
        self.a2 = None

        self.b1 = np.zeros((self.hidden_units, 1))
        self.b2 = np.zeros((self.dout, 1))
        self.W1 = 2*(np.random.random((self.din, self.hidden_units)) - 0.5)
        self.W2 = 2*(np.random.random((self.hidden_units, self.dout)) - 0.5)




    def save(self,filename):
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

    def load_mlp(filename):
        with open(filename, 'rb') as fh:
            return pickle.load(fh)
            
    def eval(self, xdata):
        ''' Evaluate the network on a set of N observations.

        xdata is a design matrix with dimensions (Din x N).
        This should return a matrix of outputs with dimension (Dout x N).
        See train_mlp.py for example usage.

        tanh activation for the hidden layer, and softmax activation for the output layer
        '''
        N = xdata.shape[1]
        outputs = np.zeros((N, self.dout))

        self.a1 = np.zeros(N, self.hidden_units)
        self.z1 = np.zeros(N, self.hidden_units)
        self.a2 = np.zeros(N, self.dout)

        #Go thru data set [0...n-1]
        for n in range(N):

            #Hidden layer

            for j in range(self.hidden_units):
                #adding offset first
                self.a1[n][j] = self.b1[j]
                #Go thru dimension [0...dim-1]
                for i in range(self.din):
                    # xdata[n][i] input value, W1[j][i] weight of the input unit towards a hidden unit
                    self.a1[n][j] += xdata[i][n] * W1[j][i]
                self.z1[n][j] = np.tanh(self.a1[n][j])


            #Output layer
            y = np.zeros(self.dout)

            softmax_sum = 0
            for k in range(self.dout):
                #adding offset first
                self.[n][k] = self.b2[k]
                #Go thru hidden layer [0...M]
                for j in range(self.hidden_units):
                    # z[j] activated value of hidden unit, W2[k][j] weight of that unit twards the output
                    self.a2[n][k] += z[j] * W2[k][j]
                softmax_sum += np.exp(self.a2[n][k])
                y[k] = np.exp(self.a2[n][k])

            y = y/softmax_sum

            outputs[n] = y

        return outputs.transpose()

    def sgd_step(self, xdata, ydata, learn_rate):
        ''' Do one step of SGD on xdata/ydata with given learning rate. 
            softmax outputs and cross-entropy error
        ''' 
        y_pred = self.eval(xdata)
        print(y_pred)

        #del_E/del_w for each output unit
        del_E = np.zeros((N, self.dout))
        #for k in range (self.dout):
        for wk in range(self.dout):
            wk = 
        pass
        
    def grad(self, xdata, ydata):
        ''' Return a tuple of the gradients of error wrt each parameter. 

        Result should be tuple of four matrices:
          (dE/dW1, dE/db1, dE/dW2, dE/db2)

        Note:  You should calculate this with backprop,
        but you might want to use finite differences for debugging.
        '''
        N = xdata.shape[1]
        dE_dW1 = np.zeros((self.din, self.hidden_units))
        dE_dW2 = np.zeros((self.hidden_units, self.dout))

        #(N * dout) computed as yk - tk for all n
        delta_k = (ydata - y_pred).transpose()
        #(N * hidden_units)
        delta_j = np.zeros((N, self.hidden_units))

        #eval to get predictions and cache of a1, z1, a2
        y_pred = self.eval(xdata)

        for n in range(N):

            #Output layer
            for k in range(self.dout):
                for j in range(self.hidden_units):
                    # del_En/del_Wjk = delta_k * (input from j to k)
                    dE_dW2[j][k] += delta_k[n][k] * self.z1[n][j]

            #Hidden unit
            for j in range(self.hidden_units):

                # delta_j = h'(aj) * SUM_over_k(Wkj * delta_k)
                prop_sum = 0 
                for k in range(self.dout):
                    prop_sum += self.W2[k][j] * delta_k[n][k]
                delta_j[n][j] = (1 - self.a1[n][j]**2) * prop_sum

                for i in range(self.din):
                    # del_En/del_Wij = delta_j * (input from i to j)
                    dE_dW1[i][j] += delta_j[n][j] * xdata[i][n]
        
        return (dE_dW1, dE_dW2)
        

