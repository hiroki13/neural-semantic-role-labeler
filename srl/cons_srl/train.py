import numpy as np

from ..utils.io_utils import say, load_conll, load_init_emb, dump_data, output_results
from ..utils.preprocess import get_id_corpus, get_phi, get_samples, get_batches, convert_data_test, shuffle
from model_api import ModelAPI


def main(argv):
    say('\nSYSTEM START')
    say('\nMODE: Training')
    say('\nRECURRENT HIDDEN UNIT: %s\n' % argv.unit)

    say('\tTRAINING\t\tBatch: %d  Epoch: %d  Parameters Save: %s' % (argv.batch, argv.epoch, argv.save))
    say('\tINITIAL EMBEDDING\t %s' % argv.init_emb)
    say('\tNETWORK STRUCTURE\tEmb Dim: %d  Hidden Dim: %d  Layers: %d' % (argv.emb, argv.hidden, argv.layer))
    say('\tOPTIMIZATION\t\tMethod: %s  Learning Rate: %f  L2 Reg: %f' % (argv.opt, argv.lr, argv.reg))

    ###############
    # Load corpus #
    ###############
    say('\n\tCorpus Preprocessing...')

    train_corpus = load_conll(argv.train_data, exclude=True)
    dev_corpus = load_conll(argv.dev_data)
    test_corpus = load_conll(argv.test_data)
    say('\tTrain Sentences: %d' % len(train_corpus))
    say('\tDev   Sentences: %d' % (len(dev_corpus) if dev_corpus else 0))
    say('\tTest  Sentences: %d' % (len(test_corpus) if test_corpus else 0))

    ##########################
    # Load initial embedding #
    ##########################
    say('\n\tInitial Embedding Loading...')
    init_emb, vocab_word = load_init_emb(init_emb=argv.init_emb)
    say('\tVocabulary Size: %d' % vocab_word.size())

    ##################
    # Create samples #
    ##################
    # samples: 1D: n_samples, 2D: (x, y)
    say('\n\tCreating Training/Dev/Test Samples...')
    tr_phi, label_dict = get_phi(train_corpus, vocab_word)
    train_samples = get_samples(tr_phi, init_emb)
    say('\tTrain Samples: %d' % len(train_samples))

    if argv.dev_data:
        dev_phi, dev_label_dict = get_phi(dev_corpus, vocab_word, label_dict)
        dev_samples = get_samples(dev_phi, init_emb)
        say('\tDev Samples: %d' % len(dev_samples))
    if argv.test_data:
        test_phi, test_label_dict = get_phi(test_corpus, vocab_word, label_dict)
        test_samples = get_samples(test_phi, init_emb)
        say('\tTest Samples: %d' % len(test_samples))

    say('\tLabel size: %d' % label_dict.size())

    ##################
    # Create batches #
    ##################
    train_samples = get_batches(train_samples, argv.batch)
    say('\tTrain Batches: %d' % len(train_samples))

    ########################
    # dump arg/vocab dicts #
    ########################
    dump_data(data=label_dict, fn='arg_dict-%d' % (label_dict.size()))
    dump_data(data=vocab_word, fn='vocab_dict-%d' % (vocab_word.size()))
    dump_data(data=init_emb, fn='emb_dict-%d' % (len(init_emb)))

    #############
    # Model API #
    #############
    model_api = ModelAPI(argv, init_emb, vocab_word, label_dict)
    model_api.set_model()
    model_api.set_train_f()
    model_api.set_test_f()

    ############
    # Training #
    ############
    say('\nTRAIN START')

    best_dev_f = 0.0
    best_test_f = 0.0
    best_epoch = -1
    flag = False

    for epoch in xrange(argv.epoch):
        ############
        # Training #
        ############
        say('\nEpoch: %d' % (epoch + 1))
        model_api.train_all(train_samples)

        ###############
        # Development #
        ###############
        if argv.dev_data:
            dev_f, predicts = model_api.predict_all(dev_samples, dev_label_dict, 'Dev')
            if best_dev_f < dev_f:
                best_dev_f = dev_f
                best_epoch = epoch

                """ Save Parameters """
                if argv.save:
                    fn = 'Model-%s.layer-%d.batch-%d.hidden-%d.reg-%f.epoch-%d' % (
                        argv.unit, argv.layer, argv.batch, argv.hidden, argv.reg, epoch)
                    dump_data(data=model_api.model, fn=fn)

                """ Output Results """
#                output_results(dev_corpus, dev_prds, label_dict, predicts,
#                               'Dev-result-%s.layer%d.batch%d.hidden%d.opt-%s.reg-%f.txt' % (
#                               argv.unit, argv.layer, argv.batch, argv.hidden, argv.opt, argv.reg))
                flag = True
            say('\t### Best Dev F Score: %f  Epoch: %d ###' % (best_dev_f, best_epoch+1))

        ########
        # Test #
        ########
        if argv.test_data:
            test_f, predicts = model_api.predict_all(test_samples, test_label_dict, 'Test')
            if flag:
                best_test_f = test_f
                flag = False
#                output_results(test_corpus, te_prds, label_dict, predicts,
#                               'Test-result-%s.layer%d.batch%d.hidden%d.opt-%s.reg-%f.txt' % (
#                               argv.unit, argv.layer, argv.batch, argv.hidden, argv.opt, argv.reg))
            if argv.test_data:
                say('\t### Best Test F Score: %f  Epoch: %d ###' % (best_test_f, best_epoch+1))
