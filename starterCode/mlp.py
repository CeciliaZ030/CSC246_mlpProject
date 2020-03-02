import numpy as np
import pickle
import random


''' An implementation of an MLP with a single layer of hidden units. '''
class MLP:
    __slots__ = ('W1', 'b1', 'a1', 'z1', 'W2', 'b2', 'din', 'dout', 'hidden_units')

    def __init__(self, din, dout, hidden_units):
        ''' Initialize a new MLP with tanh activation and softmax outputs.
        
        Params:
        din -- the dimension of the input data
        dout -- the dimension of the desired output
        hidden_units -- the number of hidden units to use

        Weights are initialized to uniform random numbers in range [-1,1).
        Biases are initalized to zero.
        
        Note: a1 and z1 can be used for caching during backprop/evaluation.
        
        '''
        self.din = din
        self.dout = dout
        self.hidden_units = hidden_units

        self.b1 = np.zeros((self.hidden_units, 1))
        self.b2 = np.zeros((self.dout, 1))
        self.W1 = 2*(np.random.random((self.hidden_units, self.din)) - 0.5)
        self.W2 = 2*(np.random.random((self.dout, self.hidden_units)) - 0.5)


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
        #Go thru data set [0...n-1]
        for n in range(N):

            #Hidden layer
            z = np.zeros(self.hidden_units)

            for j in range(self.hidden_units):
                #adding offset first
                aj = self.b1[j]
                #Go thru dimension [0...dim-1]
                for i in range(xdata.shape[0]):
                    # xni = xdata[n][i] input value, wij = W1[j][i] weight of the input unit towards a hidden unit
                    aj += xdata[i][n] * W1[j][i]
                z[j] = np.tanh(aj)

            #Output layer
            y = np.zeros(self.dout)
            softmax_sum = 0
            for k in range(self.dout):
                #adding offset first
                yk = self.b2[k]
                #Go thru hidden layer [0...M]
                for j in range(z.shape[0]):
                    # zj = z[j] activated value of hidden unit, wjk = b2[j] weight of that unit twards the output
                    yk += z[j] * W2[k][j]
                softmax_sum += np.exp(yk)
                y[k] = np.exp(yk)

            #Convert to softmax value
            for k in range(self.dout):
                y[k] = y[k]/softmax_sum

            outputs[n] = y

        return outputs.transpose()

    def sgd_step(self, xdata, ydata, learn_rate):
        ''' Do one step of SGD on xdata/ydata with given learning rate. ''' 

        #Chose a random data point
        x = np.zeros((1, xdata.shape[0]))
        print(x)
        x[0] = np.asarray(random.choices(xdata.transpose()))
        y_pred = self.eval(x.transpose())
        print(y_pred)
        #del_E/del_w for each output unit
        del_E = np.zeros(self.dout)
        #for k in range (self.dout):

        pass
        
    def grad(self, xdata, ydata):
        ''' Return a tuple of the gradients of error wrt each parameter. 

        Result should be tuple of four matrices:
          (dE/dW1, dE/db1, dE/dW2, dE/db2)

        Note:  You should calculate this with backprop,
        but you might want to use finite differences for debugging.
        '''
        pass

