__author__ = 'hiroki'


import sys
import time

import theano

from utils import load_conll, load_init_emb, get_id_samples, convert_data_test, output_results, load_data, count_correct, f_measure


def main(argv):
    print '\nSYSTEM START'
    print '\nMODE: Test'
    print '\nRECURRENT HIDDEN UNIT: %s\n' % argv.unit

    print '\tINITIAL EMBEDDING\t %s' % argv.init_emb
    print '\tNETWORK STRUCTURE\tEmb Dim: %d  Hidden Dim: %d  Layers: %d' % (argv.emb, argv.hidden, argv.layer)

    """ load corpus"""
    print '\n\tCorpus Preprocessing...'
    test_corpus = load_conll(argv.test_data)
    print '\tTest Sentences: %d' % len(test_corpus)

    """ load initial embedding file """
    print '\n\tInitial Embedding Loading...'
    init_emb, vocab_word = load_init_emb(init_emb=argv.init_emb)
    print '\tVocabulary Size: %d' % vocab_word.size()

    """ load arg dict """
    print '\n\tInitial Embedding Loading...'
    arg_dict = load_data(argv.arg_dict)
    print '\tLabel size: %d' % arg_dict.size()

    """ convert words into ids """
    print '\n\tConverting Words into IDs...'
    te_id_sents, te_id_ctx, te_marks, te_prds, test_y, test_arg_dict = get_id_samples(test_corpus,
                                                                                      vocab_word=vocab_word,
                                                                                      a_dict=arg_dict)

    """ convert formats for theano """
    print '\n\tCreating Test Samples...'
    test_sample_x, test_sample_y = convert_data_test(te_id_sents, te_prds, te_id_ctx, te_marks, test_y, init_emb)

    """ load tagger"""
    print '\nModel Loading...'
    tagger = load_data(argv.model)

    print '\nTheano Code Compiling...'

    test_model = theano.function(
        inputs=[tagger.x, tagger.d],
        outputs=[tagger.y_pred, tagger.errors],
        mode='FAST_RUN',
    )

    f, predicts = test(test_model, test_sample_x, test_sample_y, test_arg_dict, 'Test')
    output_results(test_corpus, te_prds, arg_dict, predicts,
                   'Test-result.layer%d.batch%d.hidden%d.opt-%s.reg-%f.txt' % (
                   argv.layer, argv.batch, argv.hidden, argv.opt, argv.reg))


def test(model, sample_x, sample_y,  test_arg_dict, mode):
    print '\n\t%s Index: ' % mode,
    start = time.time()

    predicts = []
    errors = []

    sample_index = 0
    for index in xrange(len(sample_x)):
        batch_x = sample_x[index]
        batch_y = sample_y[index]

        for b_index in xrange(len(batch_x)):
            sample_index += 1
            if sample_index % 100 == 0:
                print '%d' % sample_index,
                sys.stdout.flush()

            pred, error = model([batch_x[b_index]], [batch_y[b_index]])
            predicts.append(pred[0])
            errors.append(error[0])

    end = time.time()
    total, correct = count_correct(errors)
    print '\tTime: %f seconds' % (end - start)
    print '\t%s Accuracy: %f' % (mode, correct / total)

    return f_measure(predicts, sample_y, test_arg_dict), predicts

