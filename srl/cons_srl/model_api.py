import sys
import time

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
#        self.model = model(argv, max_n_agents, n_vocab, init_emb)
#        self.model.compile(c=c, r=r, a=a, y_r=y_r, y_a=y_a, n_agents=n_agents)

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

    def train_all(self, train_sample_x, train_sample_y):
        argv = self.argv
        batch = argv.batch

        print '\tIndex: ',
        start = time.time()

        losses = []
        errors = []

        sample_index = 0
        for index in xrange(len(train_sample_x)):
            batch_x = train_sample_x[index]
            batch_y = train_sample_y[index]

            for b_index in xrange(len(batch_x) / batch + 1):
                sample_index += 1
                if sample_index % 100 == 0:
                    print '%d' % sample_index,
                    sys.stdout.flush()

                sample_x = batch_x[b_index * batch: (b_index + 1) * batch]
                sample_y = batch_y[b_index * batch: (b_index + 1) * batch]

                if len(sample_x) == 0:
                    continue

                loss, error = self.train_f(sample_x, sample_y)

                losses.append(loss)
                errors.extend(error)

        end = time.time()
        avg_loss = np.mean(losses)
        total, correct = count_correct(errors)

        print '\tTime: %f seconds' % (end - start)
        print '\tAverage Negative Log Likelihood: %f' % avg_loss
        print '\tTrain Accuracy: %f' % (correct / total)

    def predict_all(self, sample_x, sample_y,  test_arg_dict, mode):
        print '\tIndex: ',
        start = time.time()

        predicts = []
        errors = []

        sample_index = 0
        for index in xrange(len(sample_x)):
            batch_x = sample_x[index]
            batch_y = sample_y[index]

            for b_index in xrange(len(batch_x)):
                sample_index += 1
                if sample_index % 1000 == 0:
                    print '%d' % sample_index,
                    sys.stdout.flush()

                pred, error = self.pred_f([batch_x[b_index]], [batch_y[b_index]])
                predicts.append(pred[0])
                errors.append(error[0])

        end = time.time()
        total, correct = count_correct(errors)
        print '\tTime: %f seconds' % (end - start)
        print '\t%s Accuracy: %f' % (mode, correct / total)

        return f_measure(predicts, sample_y, test_arg_dict), predicts

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

