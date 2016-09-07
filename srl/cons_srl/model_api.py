import sys
import time
import math

import numpy as np
import theano
import theano.tensor as T

from ..utils.io_utils import say, dump_data, load_data
from ..utils.evaluation import count_correct, f_measure
from model import Model


class ModelAPI(object):

    def __init__(self, argv, init_emb, vocab, arg_dict):
        self.argv = argv
        self.init_emb = init_emb
        self.vocab = vocab
        self.arg_dict = arg_dict

        self.model = None
        self.train_f = None
        self.pred_f = None

    def set_model(self):
        argv = self.argv

        #####################
        # Network variables #
        #####################
        x = T.ftensor3()
        d = T.imatrix()

        n_in = self.init_emb.shape[1]
        n_h = argv.hidden
        n_y = self.arg_dict.size()
        reg = argv.reg

        #################
        # Build a model #
        #################
        say('\n\nMODEL:  Unit: %s  Opt: %s' % (argv.unit, argv.opt))
        self.model = Model(unit=argv.unit, x=x, y=d, depth=argv.layer, n_in=n_in, n_h=n_h, n_y=n_y, reg=reg)

    def set_train_f(self):
        model = self.model
        self.train_f = theano.function(
            inputs=model.inputs,
            outputs=[model.nll, model.errors],
            updates=model.updates,
            mode='FAST_RUN'
        )

    def set_test_f(self):
        model = self.model
        self.pred_f = theano.function(
            inputs=model.inputs,
            outputs=[model.y_pred, model.errors],
            mode='FAST_RUN'
        )

    def train(self, c, r, a, res_vec, adr_vec, n_agents):
        nll, g_norm, pred_a, pred_r = self.train_f(c, r, a, res_vec, adr_vec, n_agents)
        return nll, g_norm, pred_a, pred_r

    def train_all(self, train_samples):
        batch_indices = range(len(train_samples))
        np.random.shuffle(batch_indices)

        print '\tIndex: ',
        start = time.time()

        losses = []
        errors = []

        for index, b_index in enumerate(batch_indices):
            if (index + 1) % 100 == 0:
                print '%d' % (index + 1),
                sys.stdout.flush()

            batch_x, batch_y = train_samples[b_index]
            loss, error = self.train_f(batch_x, batch_y)

            if math.isnan(loss):
                say('\n\nNAN: Index: %d\n' % (index + 1))
                exit()

            losses.append(loss)
            errors.extend(error)

        end = time.time()
        avg_loss = np.mean(losses)
        total, correct = count_correct(errors)

        say('\tTime: %f seconds' % (end - start))
        say('\tAverage Negative Log Likelihood: %f' % avg_loss)
        say('\tTrain Accuracy: %f' % (correct / total))

    def predict_all(self, samples, arg_dict, mode):
        print '\tIndex: ',
        start = time.time()

        predicts = []
        errors = []
        y = []

        for index, (sample_x, sample_y) in enumerate(samples):
            if (index + 1) % 1000 == 0:
                print '%d' % (index + 1),
                sys.stdout.flush()

            pred, error = self.pred_f([sample_x], [sample_y])
            predicts.append(pred[0])
            errors.append(error[0])
            y.append([sample_y])

        end = time.time()
        total, correct = count_correct(errors)
        say('\tTime: %f seconds' % (end - start))
        say('\t%s Accuracy: %f' % (mode, correct / total))

        return f_measure(predicts, y, arg_dict), predicts

    def get_pnorm_stat(self):
        lst_norms = []
        for p in self.model.params:
            vals = p.get_value(borrow=True)
            l2 = np.linalg.norm(vals)
            lst_norms.append("{:.3f}".format(l2))
        return lst_norms

    def load_model(self):
        self.model = load_data(self.argv.load)

    def save_model(self):
        argv = self.argv
        fn = 'Model-%s.batch-%d.reg-%f' % (argv.unit, argv.batch, argv.reg)
        dump_data(self.model, fn)

