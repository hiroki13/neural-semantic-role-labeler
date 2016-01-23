__author__ = 'hiroki'

import sys
import time
from copy import deepcopy
import random

import numpy as np
import theano
import theano.tensor as T

import utils
from decoupled_lstm import LSTM


random.seed(0)

PAD = u'<PAD>'
UNK = u'UNKNOWN'


def load_conll(path, file_encoding='utf-8', exclude=False):
    corpus = []

    with open(path) as f:
        sent = []
        for line in f:
            es = line.rstrip().split()
            if len(es) > 1:
                word = es[0].decode(file_encoding).lower()
                tag = es[1].decode(file_encoding)
                syn = es[2].decode(file_encoding)
                ne = es[3].decode(file_encoding)
                prd = es[4].decode(file_encoding)
                prop = []

                if len(es) > 5:
                    prop = es[5:]
                sent.append((word, tag, syn, ne, prd, prop))
            else:
                if exclude and (len(sent[0][5]) == 0 or len(sent) < 2):
                    pass
                else:
                    corpus.append(sent)
                sent = []

        if sent:
            corpus.append(sent)

    return corpus


def get_id_samples(corpus, vocab_word, a_dict=None, sort=False):
    def get_w_ids_p_indices(_sent):
        ids = []
        p_indices = []
        for w_index, w in enumerate(_sent):  # w = (form, tag, syn, ne, prd)
            w_id = vocab_word.get_id(w[0])
            if w_id is None:
                """ID for unknown word"""
                w_id = vocab_word.get_id(UNK)
            assert w_id is not None
            ids.append(w_id)

            if w[4] != '-':
                p_indices.append(w_index)

        return ids, p_indices

    def get_args(prd_i, _sent):
        _sent_args = []
        _sent_args_coarse = []

        prev = None
        for w in _sent:
            arg = w[5][prd_i]
            if arg.startswith('('):
                if arg.endswith(')'):
                    prev = arg[1:-2]
                    arg_label = 'B-' + prev
                    prev = None
                else:
                    prev = arg[1:-1]
                    arg_label = 'B-' + prev
                arg_dict.add_word(arg_label)
                _sent_args.append(arg_dict.get_id(arg_label))
                _sent_args_coarse.append((arg_dict_c.get_id('B')))
            else:
                if prev:
                    arg_label = 'I-' + prev
                    if arg.endswith(')'):
                        prev = None
                    _sent_args_coarse.append((arg_dict_c.get_id('I')))
                else:
                    arg_label = 'O'
                    _sent_args_coarse.append((arg_dict_c.get_id('O')))
                arg_dict.add_word(arg_label)
                _sent_args.append(arg_dict.get_id(arg_label))
        return _sent_args, _sent_args_coarse

    if sort:
        corpus = sorted(corpus, key=lambda sent: len(sent))

    if a_dict:
        arg_dict = deepcopy(a_dict)
    else:
        arg_dict = utils.Vocab()

    arg_dict_c = utils.Vocab()
    arg_dict_c.add_word('B')
    arg_dict_c.add_word('I')
    arg_dict_c.add_word('O')

    id_sents = []
    id_ctx = []
    marks = []
    index_prds = []
    args = []
    args_c = []
    for i, sent in enumerate(corpus):
        w_ids, prd_indices = get_w_ids_p_indices(sent)
        w_indices = [index for index in xrange(len(sent))]
        id_sents.append(w_ids)
        index_prds.append(prd_indices)

        assert len(prd_indices) == len(sent[0][5])

        sent_ctx = []
        sent_marks = []
        sent_args = []
        sent_args_c = []
        for p_i, p_index in enumerate(prd_indices):
            ctx, mark = utils.get_context(w_indices, p_index)
            sent_ctx.append(ctx)
            sent_marks.append(mark)
            s_args, s_args_c = get_args(p_i, sent)
            sent_args.append(s_args)
            sent_args_c.append(s_args_c)
        id_ctx.append(sent_ctx)
        marks.append(sent_marks)
        args.append(sent_args)
        args_c.append(sent_args_c)

    assert len(id_sents) == len(args), 'Sample x: %d\tSample y: %d' % (len(id_sents), len(args))

    return id_sents, id_ctx, marks, index_prds, args, arg_dict, args_c


def load_init_emb(init_emb):
    vocab = utils.Vocab()
    vocab.add_word(PAD)
    vocab.add_word(UNK)

    vec = {}
    with open(init_emb) as f_words:
        for line in f_words:
            line = line.strip().decode('utf-8').split()
            w = line[0]
            if w[1:-1] == UNK:
                w = UNK
            vocab.add_word(w)
            w_id = vocab.get_id(w)
            vec[w_id] = np.asarray(line[1:], dtype=theano.config.floatX)

    dim = len(line[1:])
    vec[vocab.get_id(PAD)] = np.zeros(dim, dtype=theano.config.floatX)

    emb = [[] for i in xrange(vocab.size())]
    for k, v in vec.items():
        emb[k] = v

    if vec.get(UNK) is None:
        """ averaging """
        avg = np.zeros(dim, dtype=theano.config.floatX)
        for i in xrange(len(emb)):
            avg += emb[i]
        avg_vec = avg / len(emb)
        emb[vocab.get_id(UNK)] = avg_vec

    emb = np.asarray(emb, dtype=theano.config.floatX)

    assert emb.shape[0] == vocab.size()

    return emb, vocab


def convert_data(id_sents, prds, id_ctx, marks, args, args_c, emb):
    sample_x = []
    sample_y = []
    sample_yc = []
    batch_x = []
    batch_y = []
    batch_yc = []

    pre_len = len(id_sents[0])
    for s_i in xrange(len(id_sents)):
        sent_w = [emb[w_id] for w_id in id_sents[s_i]]
        sent_prds = prds[s_i]
        sent_ctx = id_ctx[s_i]
        sent_marks = marks[s_i]
        sent_args = args[s_i]
        sent_args_c = args_c[s_i]

        """ create prd-per sample """
        p_sample = []
        for p_i, p_index in enumerate(sent_prds):
            prd = sent_w[p_index]
            ctx = []
            for w_index in sent_ctx[p_i]:
                ctx.extend(sent_w[w_index])

            mark = sent_marks[p_i]

            sent_sample = []
            for w_index, w in enumerate(sent_w):
                sample = []
                sample.extend(w)
                sample.extend(prd)
                sample.extend(ctx)
                sample.append(mark[w_index])
                sent_sample.append(sample)
            p_sample.append(sent_sample)

        sent_len = len(sent_w)

        if sent_len == pre_len:
            batch_x.extend(p_sample)
            batch_y.extend(sent_args)
            batch_yc.extend(sent_args_c)
        else:
            sample_x.append(np.asarray(batch_x, dtype=theano.config.floatX))
            sample_y.append(np.asarray(batch_y, dtype='int32'))
            sample_yc.append(np.asarray(batch_yc, dtype='int32'))

            batch_x = []
            batch_x.extend(p_sample)
            batch_y = []
            batch_y.extend(sent_args)
            batch_yc = []
            batch_yc.extend(sent_args_c)

        pre_len = sent_len

    if batch_x:
        sample_x.append(np.asarray(batch_x, dtype=theano.config.floatX))
        sample_y.append(np.asarray(batch_y, dtype='int32'))
        sample_yc.append(np.asarray(batch_yc, dtype='int32'))

    return sample_x, sample_y, sample_yc


def convert_data_test(id_sents, prds, id_ctx, marks, args, args_c, emb):
    batch_x = []
    batch_y = []
    batch_yc = []

    for s_i in xrange(len(id_sents)):
        sent_w = [emb[w_id] for w_id in id_sents[s_i]]
        sent_prds = prds[s_i]
        sent_ctx = id_ctx[s_i]
        sent_marks = marks[s_i]
        sent_args = args[s_i]
        sent_args_c = args_c[s_i]

        for p_i, p_index in enumerate(sent_prds):
            prd = sent_w[p_index]
            ctx = []
            for w_index in sent_ctx[p_i]:
                ctx.extend(sent_w[w_index])

            mark = sent_marks[p_i]
            arg = sent_args[p_i]
            arg_c = sent_args_c[p_i]

            sent_sample = []
            for w_index, w in enumerate(sent_w):
                sample = []
                sample.extend(w)
                sample.extend(prd)
                sample.extend(ctx)
                sample.append(mark[w_index])
                sent_sample.append(sample)

            batch_x.append(np.asarray([sent_sample], dtype=theano.config.floatX))
            batch_y.append(np.asarray([arg], dtype='int32'))
            batch_yc.append(np.asarray([arg_c], dtype='int32'))

    return batch_x, batch_y, batch_yc


def shuffle(sample_x, sample_y, sample_yc):
    new_x = []
    new_y = []
    new_yc = []

    indices = [i for i in xrange(len(sample_x))]
    random.shuffle(indices)

    for i in indices:
        batch_x = sample_x[i]
        batch_y = sample_y[i]
        batch_yc = sample_yc[i]
        b_indices = [j for j in xrange(len(batch_x))]
        random.shuffle(b_indices)
        new_x.append([batch_x[j] for j in b_indices])
        new_y.append([batch_y[j] for j in b_indices])
        new_yc.append([batch_yc[j] for j in b_indices])

    return new_x, new_y, new_yc


def main(argv):
    print '\nSYSTEM START\n'
    print '\nDECOUPLED TRAINING'
    print '\nUnit: %s' % argv.unit
    print 'Emb Dim: %d  Hidden Dim: %d  Optimization: %s  Layers: %d  Epoch: %d\nLearning Rate 1,2: %f %f  Batch: %d  L2 Reg: %f' % \
          (argv.emb, argv.hidden, argv.opt, argv.layer, argv.epoch, argv.lr1, argv.lr2, argv.batch, argv.reg)
    print 'Parameters to be saved: %s' % argv.save
    print 'LSTM Init Embedding: %s\tInit Layers: %s' % (argv.init_emb, argv.init_layers)

    """ load corpus"""
    print '\nCorpus Preprocessing...'
    train_corpus = load_conll(argv.train_data, exclude=True)
    test_corpus = load_conll(argv.test_data)
    print 'Train Sentences: %d\tTest Sentences: %d' % (len(train_corpus), len(test_corpus))

    """ load initial embedding file """
    print '\nInitial Embedding Loading...'
    init_emb, vocab_word = load_init_emb(init_emb=argv.init_emb)
    print 'Vocabulary Size: %d' % vocab_word.size()

    """ convert words into ids """
    print '\nConverting Words into Ids...'
    tr_id_sents, tr_id_ctx, tr_marks, tr_prds, train_y, arg_dict, train_yc = get_id_samples(train_corpus, vocab_word=vocab_word,
                                                                                  sort=True)
    te_id_sents, te_id_ctx, te_marks, te_prds, test_y, test_arg_dict, test_yc = get_id_samples(test_corpus,
                                                                                      vocab_word=vocab_word,
                                                                                      a_dict=arg_dict)
    print 'Tag size: %d' % arg_dict.size()

    """ convert formats for theano """
    print '\nCreating Training/Test Samples...'
    train_sample_x, train_sample_y, train_sample_yc = convert_data(tr_id_sents, tr_prds, tr_id_ctx, tr_marks, train_y, train_yc, init_emb)
    test_sample_x, test_sample_y, test_sample_yc = convert_data_test(te_id_sents, te_prds, te_id_ctx, te_marks, test_y, test_yc, init_emb)
    print 'Train Samples: %d' % len(train_sample_x)
    print 'Test Samples: %d' % len(test_sample_x)

    arg_dict_c = utils.Vocab()
    arg_dict_c.add_word('B')
    arg_dict_c.add_word('I')
    arg_dict_c.add_word('O')

    """symbol definition"""
    print '\nCompiling Theano Code...'
    x = T.ftensor3()
    d = T.imatrix()

    n_in = init_emb.shape[1]
    n_hidden = argv.hidden
    n_output = arg_dict_c.size()
    L2_reg = argv.reg
    batch = argv.batch

    """tagger setup"""
    tagger = LSTM(x=x, d=d, n_layers=argv.layer, op=argv.opt, n_in=n_in, n_h=n_hidden, n_y=n_output,
                  lr1=argv.lr1, lr2=argv.lr2, L2_reg=L2_reg)


    def _train(_train_sample_x, _train_sample_y):
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

        print '\nTrain START'

#        _train_sample_x, _train_sample_y = shuffle(_train_sample_x, _train_sample_y)
        for epoch in xrange(argv.epoch):
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

            print '\tTime: %f seconds' % (end - start)
            print '\tAverage Negative Log Likelihood: %f' % avg_loss

            total = 0.0
            correct = 0
            for sent in errors:
                total += len(sent)
                for y_pred in sent:
                    if y_pred == 0:
                        correct += 1

            print '\tTrain Accuracy: %f' % (correct / total)

            _test(test_model, epoch)

    def _test(model, epoch=0):
        print '\tTest Index: ',
        start = time.time()

        predicts = []
        errors = []

        sample_index = 0
        for index in xrange(len(test_sample_x)):
            batch_x = test_sample_x[index]
            batch_y = test_sample_yc[index]

            for b_index in xrange(len(batch_x)):
                sample_index += 1
                if sample_index % 100 == 0:
                    print '%d' % sample_index,
                    sys.stdout.flush()

                sample_x = batch_x[b_index]
                sample_y = batch_y[b_index]

                pred, error = model([sample_x], [sample_y])

                predicts.append(pred[0])
                errors.append(error[0])

        end = time.time()
        print '\tTime: %f seconds' % (end - start)

        total = 0.0
        correct = 0
        for sent in errors:
            total += len(sent)
            for y_pred in sent:
                if y_pred == 0:
                    correct += 1
        print '\tTest Accuracy: %f' % (correct / total)
        f_measure_c(predicts, test_sample_yc, arg_dict_c)
        utils.output_results(test_corpus, te_prds, arg_dict, predicts,
                             'result.layer%d.batch%d.emb%d.opt-%s.lr1-%f.lr2-%f.reg-%f.epoch%d.txt' % (
                                 argv.layer, argv.batch, argv.emb, argv.opt, argv.lr1, argv.lr2, argv.reg, epoch))

    _train(train_sample_x, train_sample_yc)


def count_spans(spans):
    total = 0
    for span in spans:
        if not span[0].startswith('C'):
            total += 1
    return total


def f_measure(predicts, answers, arg_dict):
    def get_spans(sent):
        spans = []
        span = []
        for w_i, a_id in enumerate(sent):
            label = arg_dict.get_word(a_id)
            if label.startswith('B-'):
                if span and span[0][0] != 'V':
                    spans.append(span)
                span = [label[2:], w_i, w_i]
            elif label.startswith('I-'):
                if span:
                    if label[2:] == span[0]:
                        span[2] = w_i
                    else:
                        if span[0][0] != 'V':
                            spans.append(span)
                        span = [label[2:], w_i, w_i]
                else:
                    span = [label[2:], w_i, w_i]
            else:
                if span and span[0][0] != 'V':
                    spans.append(span)
                span = []
        if span:
            spans.append(span)
        return spans

    p_total = 0.
    r_total = 0.
    correct = 0.
    for i in xrange(len(predicts)):
        ys = predicts[i]
        ds = answers[i][0]
        y_spans = get_spans(ys)
        d_spans = get_spans(ds)
        p_total += count_spans(y_spans)
        r_total += count_spans(d_spans)

        for y_span in y_spans:
            if y_span[0].startswith('C'):
                continue
            if y_span in d_spans:
                correct += 1.

    if p_total > 0:
        p = correct / p_total
    else:
        p = 0.
    if r_total > 0:
        r = correct / r_total
    else:
        r = 0.
    if p + r > 0:
        f = (2 * p * r) / (p + r)
    else:
        f = 0.
    print '\tProps: %d\tP total: %f\tR total: %f\tCorrect: %f' % (len(predicts), p_total, r_total, correct)
    print '\tPrecision: %f\tRecall: %f\tF1: %f' % (p, r, f)


def f_measure_c(predicts, answers, arg_dict):
    def get_spans(sent):
        spans = []
        span = []
        for w_i, a_id in enumerate(sent):
            label = arg_dict.get_word(a_id)
            if label.startswith('B'):
                if span:
                    spans.append(span)
                span = [w_i, w_i]
            elif label.startswith('I'):
                if span:
                    span[1] = w_i
                else:
                    span = [w_i, w_i]
            else:
                if span:
                    spans.append(span)
                span = []
        if span:
            spans.append(span)
        return spans

    p_total = 0.
    r_total = 0.
    correct = 0.
    for i in xrange(len(predicts)):
        ys = predicts[i]
        ds = answers[i][0]
        y_spans = get_spans(ys)
        d_spans = get_spans(ds)
        p_total += len(y_spans)
        r_total += len(d_spans)

        for y_span in y_spans:
#            if y_span[0].startswith('C'):
#                continue
            if y_span in d_spans:
                correct += 1.

    if p_total > 0:
        p = correct / p_total
    else:
        p = 0.
    if r_total > 0:
        r = correct / r_total
    else:
        r = 0.
    if p + r > 0:
        f = (2 * p * r) / (p + r)
    else:
        f = 0.
    print '\tProps: %d\tP total: %f\tR total: %f\tCorrect: %f' % (len(predicts), p_total, r_total, correct)
    print '\tPrecision: %f\tRecall: %f\tF1: %f' % (p, r, f)
