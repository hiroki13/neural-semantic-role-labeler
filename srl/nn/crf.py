import theano
import theano.tensor as T

from srl.nn.nn_utils import logsumexp, sample_weights, relu


class CRF(object):

    def __init__(self, n_i, n_h):
        self.W = theano.shared(sample_weights(n_i, n_h))
        self.W_t = theano.shared(sample_weights(n_h, n_h))
        self.BOS = theano.shared(sample_weights(n_h))
        self.params = [self.W, self.W_t, self.BOS]

    def dot(self, x):
        return relu(T.dot(x, self.W))

    def y_prob(self, h, y, batch):
        """
        :param h: 1D: n_words, 2D: Batch, 3D: n_y
        :param y: 1D: n_words, 2D: Batch
        :return: gradient of cross entropy: 1D: Batch
        """
        y_score0 = self.BOS[y[0]] + h[0][T.arange(batch), y[0]]  # 1D: Batch
        z_score0 = self.BOS + h[0]  # 1D: Batch, 2D: n_y

        [_, y_scores, z_scores], _ = theano.scan(fn=self.forward_step,
                                                 sequences=[h[1:], y[1:]],
                                                 outputs_info=[y[0], y_score0, z_score0],
                                                 non_sequences=[self.W_t, batch])

        y_score = y_scores[-1]
        z_score = logsumexp(z_scores[-1], axis=1).flatten()

        return y_score - z_score

    def forward_step(self, e_t, d_t, d_prev, d_score_prev, z_scores_prev, trans, batch):
        """
        :param e_t: 1D: Batch, 2D: n_y
        :param d_t: 1D: Batch
        :param d_prev: 1D: Batch
        :param d_score_prev: 1D: Batch
        :param z_scores_prev: 1D: Batch, 2D: n_y
        :param trans: 1D: n_y, 2D, n_y
        """
        d_score_t = d_score_prev + trans[d_t, d_prev] + e_t[T.arange(batch), d_t]  # 1D: Batch
        z_sum = z_scores_prev.dimshuffle(0, 'x', 1) + trans  # 1D: Batch, 2D: n_y, 3D: n_y
        z_scores_t = logsumexp(z_sum, axis=2).reshape(e_t.shape) + e_t  # 1D: Batch, 2D: n_y
        return d_t, d_score_t, z_scores_t

    def forward(self, e_t, score_prev, trans):
        """
        :param e_t: 1D: Batch, 2D: n_y
        :param scores_prev: 1D: Batch, 2D: n_y
        :param trans: 1D: n_y, 2D, n_y
        """
        score = score_prev.dimshuffle(0, 'x', 1) + trans + e_t.dimshuffle(0, 1, 'x')
        max_scores_t, max_nodes_t = T.max_and_argmax(score, axis=2)
        return max_scores_t, T.cast(max_nodes_t, dtype='int32')

    def backward(self, nodes_t, max_node_t, batch):
        return nodes_t[T.arange(batch), max_node_t]

    def vitabi(self, emit, batch):
        scores0 = self.BOS + emit[0]
        [max_scores, max_nodes], _ = theano.scan(fn=self.forward,
                                                 sequences=[emit[1:]],
                                                 outputs_info=[scores0, None],
                                                 non_sequences=self.W_t)
        max_last_node = T.cast(T.argmax(max_scores[-1], axis=1), dtype='int32')

        nodes, _ = theano.scan(fn=self.backward,
                               sequences=max_nodes[::-1],
                               outputs_info=max_last_node,
                               non_sequences=batch)

        return T.concatenate([nodes[::-1].dimshuffle(1, 0), max_last_node.dimshuffle((0, 'x'))], 1)

