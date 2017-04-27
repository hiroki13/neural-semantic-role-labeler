import theano
import theano.tensor as T

from ..nn.nn_utils import logsumexp, sample_weights


class CRF(object):

    def __init__(self, n_i, n_h):
        self.W = theano.shared(sample_weights(n_i, n_h))
        self.W_t = theano.shared(sample_weights(n_h, n_h))
        self.BOS = theano.shared(sample_weights(n_h))
        self.b = theano.shared(sample_weights(n_h))
        self.params = [self.W, self.W_t, self.BOS, self.b]

    def dot(self, x):
        return T.dot(x, self.W) + self.b

    def y_prob(self, h, y, batch):
        """
        :param h: 1D: n_words, 2D: Batch, 3D: n_y
        :param y: 1D: n_words, 2D: Batch
        :return: gradient of cross entropy: 1D: Batch
        """
        batch_index = T.arange(batch)
        z_score0 = self.BOS + h[0]  # 1D: batch, 2D: n_y
        y_score0 = z_score0[batch_index, y[0]]  # 1D: batch

        [_, y_scores, z_scores], _ = theano.scan(fn=self._forward_step,
                                                 sequences=[h[1:], y[1:]],
                                                 outputs_info=[y[0], y_score0, z_score0],
                                                 non_sequences=[self.W_t, batch_index])

        y_score = y_scores[-1]
        z_score = logsumexp(z_scores[-1], axis=1).flatten()

        return y_score - z_score

    @staticmethod
    def _forward_step(h_t, y_t, y_prev, y_score_prev, z_score_prev, trans, batch_index):
        """
        :param h_t: 1D: Batch, 2D: n_y
        :param y_t: 1D: Batch
        :param y_prev: 1D: Batch
        :param y_score_prev: 1D: Batch
        :param z_score_prev: 1D: Batch, 2D: n_y
        :param trans: 1D: n_y, 2D, n_y
        """
        y_score_t = y_score_prev + trans[y_t, y_prev] + h_t[batch_index, y_t]  # 1D: Batch
        z_sum = z_score_prev.dimshuffle(0, 'x', 1) + trans  # 1D: Batch, 2D: n_y, 3D: n_y
        z_score_t = logsumexp(z_sum, axis=2).reshape(h_t.shape) + h_t  # 1D: Batch, 2D: n_y
        return y_t, y_score_t, z_score_t

    def vitabi(self, h, batch):
        score0 = self.BOS + h[0]
        [y_scores, y_nodes], _ = theano.scan(fn=self._vitabi_forward,
                                             sequences=[h[1:]],
                                             outputs_info=[score0, None],
                                             non_sequences=self.W_t)

        y_hat_last = T.cast(T.argmax(y_scores[-1], axis=1), dtype='int32')
        y_hat, _ = theano.scan(fn=self._vitabi_backward,
                               sequences=y_nodes[::-1],
                               outputs_info=y_hat_last,
                               non_sequences=T.arange(batch))

        return T.concatenate([y_hat[::-1].dimshuffle(1, 0), y_hat_last.dimshuffle((0, 'x'))], 1)

    @staticmethod
    def _vitabi_forward(e_t, score_prev, trans):
        """
        :param e_t: 1D: Batch, 2D: n_y
        :param score_prev: 1D: Batch, 2D: n_y
        :param trans: 1D: n_y, 2D, n_y
        """
        score = score_prev.dimshuffle(0, 'x', 1) + trans + e_t.dimshuffle(0, 1, 'x')
        max_scores_t, max_nodes_t = T.max_and_argmax(score, axis=2)
        return max_scores_t, T.cast(max_nodes_t, dtype='int32')

    @staticmethod
    def _vitabi_backward(y_nodes_t, y_hat_t, batch_index):
        return y_nodes_t[batch_index, y_hat_t]

