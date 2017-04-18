import theano
import theano.tensor as T

from nn_utils import tanh, sample_weights


class BaseUnit(object):

    def __init__(self, n_i=32, n_h=32, activation=tanh):
        self.activation = activation
        self.W = theano.shared(sample_weights(n_i, n_h))
        self.params = [self.W]

    def dot(self, x):
        return self.activation(T.dot(x, self.W))


