import theano
import theano.tensor as T

from srl.nn.crf import CRF
from ..nn.nn_utils import L2_sqr
from ..nn.optimizers import adam
from ..nn.rnn import GRU, GRU2, LSTM, Layer
from ..utils.io_utils import say


class Model2(object):

    def __init__(self, unit, x, y, depth, n_in, n_h, n_y, reg=0.0001):

        self.x = x  # 1D: batch_size, 2D: n_words, 3D: n_fin
        self.y = y  # 1D: batch_size, 2D: n_words
        self.inputs = [x, y]
        self.layers = []
        self.params = []

        n_fin = n_in * 7 + 1
        batch = T.cast(y.shape[0], dtype='int32')

        ################
        # Set networks #
        ################
        self.set_layers(unit, n_fin, n_h, n_y, depth)
        self.set_params()

        #############
        # Computing #
        #############
        x, h = self.mid_layer(x)
        h = self.output_layer(x, h)

        ##########
        # Scores #
        ##########
        self.p_y = self.get_y_scores(h, y.dimshuffle(1, 0), batch)
        self.y_pred = self.get_scores(h, batch)
        self.errors = T.neq(self.y_pred, y)

        ############
        # Training #
        ############
        self.nll = - T.mean(self.p_y)
        self.cost = self.nll + reg * L2_sqr(self.params) / 2.
        self.g = T.grad(self.cost, self.params)
        self.updates = adam(self.params, self.g)

    def set_layers(self, unit, n_fin, n_h, n_y, depth):
        if unit.lower() == 'gru':
            layer = GRU
        else:
            layer = LSTM

        for i in xrange(depth):
            if i == 0:
                self.layers.append(layer(n_i=n_fin, n_h=n_h))
            else:
                self.layers.append(layer(n_i=n_h * 2, n_h=n_h))

        self.layers.append(CRF(n_i=n_h * 2, n_h=n_y))
        say('Hidden Layer: %d' % (len(self.layers) - 1))

    def set_params(self):
        for l in self.layers:
            self.params += l.params
        say("num of parameters: {}\n".format(sum(len(x.get_value(borrow=True).ravel()) for x in self.params)))

    def mid_layer(self, x):
        x = x.dimshuffle(1, 0, 2)

        for i in xrange(len(self.layers)-1):
            layer = self.layers[i]
            if i == 0:
                x = layer.dot(x)
                h0 = T.zeros_like(x[0], dtype=theano.config.floatX)
            else:
                x = T.concatenate([x, h], 2)
                x = layer.dot(x)[::-1]
#                h0 = x[0]
                h0 = h[-1]

            h = layer.forward_all(x, h0)

        return x, h

    def output_layer(self, x, h):
        crf = self.layers[-1]

        h = T.concatenate([x, h], axis=2)
        h = crf.dot(h)

        if (len(self.layers) - 1) % 2 == 0:
            h = h[::-1]

        return h

    def get_scores(self, h, batch):
        crf = self.layers[-1]
        return crf.vitabi(h, batch)

    def get_y_scores(self, h, y, batch):
        crf = self.layers[-1]
        return crf.y_prob(h, y, batch)


class Model(object):

    def __init__(self, unit, x, y, depth, n_in, n_h, n_y, reg=0.0001):

        self.x = x  # 1D: batch_size, 2D: n_words, 3D: n_fin
        self.y = y  # 1D: batch_size, 2D: n_words
        self.inputs = [x, y]
        self.layers = []
        self.params = []

        n_fin = n_in * 7 + 1
        batch = T.cast(x.shape[0], dtype='int32')

        ################
        # Set networks #
        ################
        self.set_layers(unit, n_fin, n_h, n_y, depth)
        self.set_params()

        #############
        # Computing #
        #############
        h1 = self.input_layer(x)
        h2 = self.mid_layer(h1)
        o = self.output_layer(h2)

        ##########
        # Scores #
        ##########
        self.p_y = self.get_y_scores(o, y.dimshuffle(1, 0), batch)
        self.y_pred = self.get_scores(o, batch)
        self.errors = T.neq(self.y_pred, y)

        ############
        # Training #
        ############
        self.nll = - T.mean(self.p_y)
        self.cost = self.nll + reg * L2_sqr(self.params) / 2.
        self.g = T.grad(self.cost, self.params)
        self.updates = adam(self.params, self.g)

    def set_layers(self, unit, n_fin, n_h, n_y, depth):
        if unit.lower() == 'gru':
            layer = GRU2
        else:
            layer = LSTM

        self.layers.append(Layer(n_i=n_fin, n_h=n_h))
        for i in xrange(depth):
            self.layers.append(layer(n_i=n_h, n_h=n_h))

        self.layers.append(CRF(n_i=n_h, n_h=n_y))
        say('Hidden Layer: %d' % (len(self.layers) - 2))

    def set_params(self):
        for l in self.layers:
            self.params += l.params
        say("num of parameters: {}\n".format(sum(len(x.get_value(borrow=True).ravel()) for x in self.params)))

    def input_layer(self, x):
        return self.layers[0].dot(x.dimshuffle(1, 0, 2))

    def mid_layer(self, x):
        h0 = T.zeros_like(x[0], dtype=theano.config.floatX)
        for layer in self.layers[1:-1]:
            h = layer.forward_all(x, h0)
            x = (x + h)[::-1]
            h0 = h[-1]
        if (len(self.layers) - 2) % 2 == 1:
            x = x[::-1]
        return x

    def output_layer(self, h):
        crf = self.layers[-1]
        return crf.dot(h)

    def get_scores(self, h, batch):
        crf = self.layers[-1]
        return crf.vitabi(h, batch)

    def get_y_scores(self, h, y, batch):
        crf = self.layers[-1]
        return crf.y_prob(h, y, batch)



