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
        self.W1 = 2*(np.random.random_sample((self.din, self.hidden_units)) - 0.5)
        self.W2 = 2*(np.random.random_sample((self.hidden_units, self.dout)) - 0.5)

        print("init")
        print(self.W1, self.W2)
        print(self.din, self.hidden_units, self.dout)




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

        print("eval re-init cache")
        self.a1 = np.zeros((N, self.hidden_units))
        self.z1 = np.zeros((N, self.hidden_units))
        self.a2 = np.zeros((N, self.dout))
        print(self.b1, self.W1)
        #Go thru data set [0...n-1]
        for n in range(N):

            #Hidden layer

            for j in range(self.hidden_units):
                #adding offset first
                self.a1[n][j] = self.b1[j]
                #Go thru dimension [0...dim-1]
                for i in range(self.din):
                    # xdata[n][i] input value, W1[j][i] weight of the input unit towards a hidden unit
                    self.a1[n][j] += xdata[i][n] * self.W1[i][j]
                self.z1[n][j] = np.tanh(self.a1[n][j])


            #Output layer
            y = np.zeros(self.dout)

            softmax_sum = 0
            for k in range(self.dout):
                #adding offset first
                self.a2[n][k] = self.b2[k]
                #Go thru hidden layer [0...M]
                for j in range(self.hidden_units):
                    # z1[n][j] activated value of hidden unit, W2[k][j] weight of that unit twards the output
                    self.a2[n][k] += self.z1[n][j] * self.W2[j][k]
                softmax_sum += np.exp(self.a2[n][k])
                y[k] = np.exp(self.a2[n][k])

            y = y/softmax_sum
            outputs[n] = y

        #print("eval")
        print(outputs)
        return outputs.transpose()

    def sgd_step(self, xdata, ydata, learn_rate):
        ''' Do one step of SGD on xdata/ydata with given learning rate. 
            softmax outputs and cross-entropy error
        ''' 
        print("sgd_step call grad")
        dE_dW1, dE_db1, dE_dW2, dE_db2 = self.grad(xdata, ydata)


        self.W1 = [[self.W1[i][j] + learn_rate * dE_dW1[i][j] 
                            for j in range(self.hidden_units)] 
                            for i in range(self.din)]

        self.b1 =  [self.b1[i] + learn_rate * dE_db1[i] 
                            for i in range(self.hidden_units)]

        self.W2 = [[self.W2[i][j] + learn_rate * dE_dW2[i][j] 
                            for j in range(self.dout)] 
                            for i in range(self.hidden_units)]

        self.b2 =  [self.b2[i] + learn_rate * dE_db2[i] 
                            for i in range(self.dout)]


        
    def grad(self, xdata, ydata):
        ''' Return a tuple of the gradients of error wrt each parameter. 

        Result should be tuple of four matrices:
          (dE/dW1, dE/db1, dE/dW2, dE/db2)

        Note:  You should calculate this with backprop,
        but you might want to use finite differences for debugging.
        '''
        N = xdata.shape[1]
        dE_dW1 = np.zeros((self.din, self.hidden_units))
        dE_db1 = np.zeros((self.hidden_units))
        dE_dW2 = np.zeros((self.hidden_units, self.dout))
        dE_db2 = np.zeros((self.dout))

        #eval to get predictions and cache of a1, z1, a2
        print("grad call eval")
        y_pred = self.eval(xdata)

        #(N * dout) computed as yk - tk for all n
        delta_k = (ydata - y_pred).transpose()
        # print("delta_k shape: ", delta_k.shape)
        # print("dE_db2 shape: ", dE_db2.shape)
        # print(ydata, y_pred)

        #(N * hidden_units)
        delta_j = np.zeros((N, self.hidden_units))


        for n in range(N):

            #Output layer

            for k in range(self.dout):

                # del_En/del_b2 = delta_k
                dE_db2[k] += delta_k[n][k]
                #print(delta_k[n][k])

                for j in range(self.hidden_units):
                    # del_En/del_Wjk = delta_k * (input from j to k)
                    dE_dW2[j][k] += delta_k[n][k] * self.z1[n][j]

            #Hidden unit

            for j in range(self.hidden_units):
                # del_En/del_b1 = delta_j
                dE_db1[j] += delta_j[n][j]

                # delta_j = h'(aj) * SUM_over_k(Wkj * delta_k)
                prop_sum = 0 
                for k in range(self.dout):
                    prop_sum += self.W2[j][k] * delta_k[n][k]

                print(self.a1)
                delta_j[n][j] = (1 - self.a1[n][j]**2) * prop_sum

                for i in range(self.din):
                    # del_En/del_Wij = delta_j * (input from i to j)
                    dE_dW1[i][j] += delta_j[n][j] * xdata[i][n]
        
        print(dE_db2)
        return (dE_dW1/N, dE_db1/N, dE_dW2/N, dE_db2/N)
        

