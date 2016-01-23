__author__ = 'hiroki'

import sys
import time

import numpy as np
import theano
import theano.tensor as T

from rnn import RNN
from test import test
from utils import load_conll, load_init_emb, get_id_samples, convert_data, convert_data_test, shuffle, count_correct, dump_data, output_results


def main(argv):
    print '\nSYSTEM START'
    print '\nMODE: Training'
    print '\nRECURRENT HIDDEN UNIT: %s\n' % argv.unit

    print '\tTRAINING\t\tBatch: %d  Epoch: %d  Parameters Save: %s' % (argv.batch, argv.epoch, argv.save)
    print '\tINITIAL EMBEDDING\t %s' % argv.init_emb
    print '\tNETWORK STRUCTURE\tEmb Dim: %d  Hidden Dim: %d  Layers: %d' % (argv.emb, argv.hidden, argv.layer)
    print '\tOPTIMIZATION\t\tMethod: %s  Learning Rate: %f %f  L2 Reg: %f' % (argv.opt, argv.lr1, argv.lr2, argv.reg)

    """ load corpus"""
    print '\n\tCorpus Preprocessing...'

    train_corpus = load_conll(argv.train_data, exclude=True)
    print '\tTrain Sentences: %d' % len(train_corpus)

    if argv.dev_data:
        dev_corpus = load_conll(argv.dev_data)
        print '\tDev   Sentences: %d' % len(dev_corpus)

    if argv.test_data:
        test_corpus = load_conll(argv.test_data)
        print '\tTest  Sentences: %d' % len(test_corpus)

    """ load initial embedding file """
    print '\n\tInitial Embedding Loading...'
    init_emb, vocab_word = load_init_emb(init_emb=argv.init_emb)
    print '\tVocabulary Size: %d' % vocab_word.size()

    """ convert words into ids """
    print '\n\tConverting Words into IDs...'

    tr_id_sents, tr_id_ctx, tr_marks, tr_prds, train_y, arg_dict = get_id_samples(train_corpus, vocab_word=vocab_word,
                                                                                  sort=True)

    if argv.dev_data:
        dev_id_sents, dev_id_ctx, dev_marks, dev_prds, dev_y, dev_arg_dict =\
            get_id_samples(dev_corpus, vocab_word=vocab_word, a_dict=arg_dict)
    if argv.test_data:
        te_id_sents, te_id_ctx, te_marks, te_prds, test_y, test_arg_dict =\
            get_id_samples(test_corpus, vocab_word=vocab_word, a_dict=arg_dict)

    print '\tLabel size: %d' % arg_dict.size()
    dump_data(data=arg_dict, fn='arg_dict-%d' % (arg_dict.size()))

    """ convert formats for theano """
    print '\n\tCreating Training/Dev/Test Samples...'

    train_sample_x, train_sample_y = convert_data(tr_id_sents, tr_prds, tr_id_ctx, tr_marks, train_y, init_emb)
    print '\tTrain Samples: %d' % len(train_sample_x)

    if argv.dev_data:
        dev_sample_x, dev_sample_y = convert_data_test(dev_id_sents, dev_prds, dev_id_ctx, dev_marks, dev_y, init_emb)
        print '\tDev Samples: %d' % len(dev_sample_x)

    if argv.test_data:
        test_sample_x, test_sample_y = convert_data_test(te_id_sents, te_prds, te_id_ctx, te_marks, test_y, init_emb)
        print '\tTest Samples: %d' % len(test_sample_x)

    """symbol definition"""
    x = T.ftensor3()
    d = T.imatrix()

    n_in = init_emb.shape[1]
    n_h = argv.hidden
    n_y = arg_dict.size()
    reg = argv.reg
    batch = argv.batch

    """ Model Setup """
    print '\nTheano Code Compiling...'

    tagger = RNN(unit=argv.unit, x=x, d=d, n_layers=argv.layer, n_in=n_in, n_h=n_h, n_y=n_y, reg=reg)

    train_model = theano.function(
        inputs=[x, d],
        outputs=[tagger.nll, tagger.errors],
        updates=tagger.updates,
        mode='FAST_RUN'
    )

    test_model = theano.function(
        inputs=[x, d],
        outputs=[tagger.y_pred, tagger.errors],
        mode='FAST_RUN'
    )

    """ Training """
    print '\nTRAIN START'

    best_dev_f = 0.0
    best_test_f = 0.0
    best_epoch = -1
    flag = False

    for epoch in xrange(argv.epoch):
        _train_sample_x, _train_sample_y = shuffle(train_sample_x, train_sample_y)

        print '\nEpoch: %d' % (epoch + 1)
        print '\tIndex: ',
        start = time.time()

        losses = []
        errors = []

        sample_index = 0
        for index in xrange(len(train_sample_x)):
            batch_x = _train_sample_x[index]
            batch_y = _train_sample_y[index]

            for b_index in xrange(len(batch_x) / batch + 1):
                sample_index += 1
                if sample_index % 100 == 0:
                    print '%d' % sample_index,
                    sys.stdout.flush()

                sample_x = batch_x[b_index * batch: (b_index + 1) * batch]
                sample_y = batch_y[b_index * batch: (b_index + 1) * batch]

                if len(sample_x) == 0:
                    continue

                loss, error = train_model(sample_x, sample_y)

                losses.append(loss)
                errors.extend(error)

        end = time.time()
        avg_loss = np.mean(losses)
        total, correct = count_correct(errors)

        print '\tTime: %f seconds' % (end - start)
        print '\tAverage Negative Log Likelihood: %f' % avg_loss
        print '\tTrain Accuracy: %f' % (correct / total)

        """ Check model performance """
        if argv.dev_data:
            dev_f, predicts = test(test_model, dev_sample_x, dev_sample_y, dev_arg_dict, 'Dev')
            if best_dev_f < dev_f:
                best_dev_f = dev_f
                best_epoch = epoch

                """ Save Parameters """
                if argv.save:
                    fn = 'Layer-%d_Dim-%d_Batch-%d_Hidden-%d_Reg-%f_Epoch-%d' % (
                        argv.layer, argv.hidden, argv.batch, argv.hidden, argv.reg, epoch)
                    dump_data(data=tagger, fn=fn)

                """ Output Results """
                output_results(dev_corpus, dev_prds, arg_dict, predicts,
                               'Dev-result.layer%d.batch%d.hidden%d.opt-%s.reg-%f.epoch%d.txt' % (
                               argv.layer, argv.batch, argv.hidden, argv.opt, argv.reg, epoch))
                flag = True
            print '\t### Best Dev F Score: %f  Epoch: %d ###' % (best_dev_f, best_epoch+1)

        if argv.test_data:
            test_f, predicts = test(test_model, test_sample_x, test_sample_y, test_arg_dict, 'Test')
            if flag:
                best_test_f = test_f
                flag = False
                output_results(test_corpus, te_prds, arg_dict, predicts,
                               'Test-result.layer%d.batch%d.hidden%d.opt-%s.reg-%f.epoch%d.txt' % (
                               argv.layer, argv.batch, argv.hidden, argv.opt, argv.reg, epoch))
            if argv.dev_data:
                print '\t### Best Test F Score: %f  Epoch: %d ###' % (best_test_f, best_epoch+1)
