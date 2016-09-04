from ..utils.io_utils import say, load_conll, load_init_emb, get_id_samples, dump_data, output_results
from ..utils.preprocess import convert_data, convert_data_test, shuffle
from model_api import ModelAPI


def main(argv):
    say('\nSYSTEM START')
    say('\nMODE: Training')
    say('\nRECURRENT HIDDEN UNIT: %s\n' % argv.unit)

    say('\tTRAINING\t\tBatch: %d  Epoch: %d  Parameters Save: %s' % (argv.batch, argv.epoch, argv.save))
    say('\tINITIAL EMBEDDING\t %s' % argv.init_emb)
    say('\tNETWORK STRUCTURE\tEmb Dim: %d  Hidden Dim: %d  Layers: %d' % (argv.emb, argv.hidden, argv.layer))
    say('\tOPTIMIZATION\t\tMethod: %s  Learning Rate: %f %f  L2 Reg: %f' % (argv.opt, argv.lr1, argv.lr2, argv.reg))

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

    ##########################
    # Convert words into ids #
    ##########################
    say('\n\tConverting Words into IDs...')

    tr_id_sents, tr_id_ctx, tr_marks, tr_prds, train_y, arg_dict = get_id_samples(train_corpus, vocab_word=vocab_word,
                                                                                  sort=True)

    if argv.dev_data:
        dev_id_sents, dev_id_ctx, dev_marks, dev_prds, dev_y, dev_arg_dict =\
            get_id_samples(dev_corpus, vocab_word=vocab_word, a_dict=arg_dict)
    if argv.test_data:
        te_id_sents, te_id_ctx, te_marks, te_prds, test_y, test_arg_dict =\
            get_id_samples(test_corpus, vocab_word=vocab_word, a_dict=arg_dict)

    print '\tLabel size: %d' % arg_dict.size()

    """ dump arg/vocab dicts """
    dump_data(data=arg_dict, fn='arg_dict-%d' % (arg_dict.size()))
    dump_data(data=vocab_word, fn='vocab_dict-%d' % (vocab_word.size()))
    dump_data(data=init_emb, fn='emb_dict-%d' % (len(init_emb)))

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

    #############
    # Model API #
    #############
    model_api = ModelAPI(argv, init_emb, vocab_word, arg_dict)
    model_api.set_model()
    model_api.set_train_f()
    model_api.set_test_f()

    ############
    # Training #
    ############
    print '\nTRAIN START'

    best_dev_f = 0.0
    best_test_f = 0.0
    best_epoch = -1
    flag = False

    for epoch in xrange(argv.epoch):
        ############
        # Training #
        ############
        print '\nEpoch: %d' % (epoch + 1)
        train_sample_x, train_sample_y = shuffle(train_sample_x, train_sample_y)
        model_api.train_all(train_sample_x, train_sample_y)

        ###############
        # Development #
        ###############
        if argv.dev_data:
            dev_f, predicts = model_api.predict_all(dev_sample_x, dev_sample_y, dev_arg_dict, 'Dev')
            if best_dev_f < dev_f:
                best_dev_f = dev_f
                best_epoch = epoch

                """ Save Parameters """
                if argv.save:
                    fn = 'Model-%s.layer-%d.batch-%d.hidden-%d.reg-%f.epoch-%d' % (
                        argv.unit, argv.layer, argv.batch, argv.hidden, argv.reg, epoch)
                    dump_data(data=model_api.model, fn=fn)

                """ Output Results """
                output_results(dev_corpus, dev_prds, arg_dict, predicts,
                               'Dev-result-%s.layer%d.batch%d.hidden%d.opt-%s.reg-%f.txt' % (
                               argv.unit, argv.layer, argv.batch, argv.hidden, argv.opt, argv.reg))
                flag = True
            print '\t### Best Dev F Score: %f  Epoch: %d ###' % (best_dev_f, best_epoch+1)

        ########
        # Test #
        ########
        if argv.test_data:
            test_f, predicts = model_api.predict_all(test_sample_x, test_sample_y, test_arg_dict, 'Test')
            if flag:
                best_test_f = test_f
                flag = False
                output_results(test_corpus, te_prds, arg_dict, predicts,
                               'Test-result-%s.layer%d.batch%d.hidden%d.opt-%s.reg-%f.txt' % (
                               argv.unit, argv.layer, argv.batch, argv.hidden, argv.opt, argv.reg))
            if argv.test_data:
                print '\t### Best Test F Score: %f  Epoch: %d ###' % (best_test_f, best_epoch+1)
