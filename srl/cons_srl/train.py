from ..utils.io_utils import say
from ..utils.loader import load_conll, load_init_emb
from ..utils.saver import dump_data, save_predicted_prop
from ..utils.preprocess import get_x, get_y, concat_x_y, get_vocab_label, get_samples, get_batches
from ..utils.evaluation import show_f1_history
from model_api import ModelAPI


def get_dataset(argv):
    say('\n\tCorpus Preprocessing...')

    train_corpus = load_conll(argv.train_data, data_size=argv.data_size)
    dev_corpus = load_conll(argv.dev_data, data_size=argv.data_size)
    test_corpus = load_conll(argv.test_data, data_size=argv.data_size)

    say('\tTrain Sentences: %d' % len(train_corpus))
    say('\tDev   Sentences: %d' % (len(dev_corpus) if dev_corpus else 0))
    say('\tTest  Sentences: %d' % (len(test_corpus) if test_corpus else 0))

    return train_corpus, dev_corpus, test_corpus


def get_init_emb(argv):
    say('\n\tInitial Embedding Loading...')
    init_emb, vocab_word = load_init_emb(init_emb=argv.init_emb)
    say('\tVocabulary Size: %d' % vocab_word.size())
    return init_emb, vocab_word


def create_samples(corpus, vocab_word, vocab_label=None, init_emb=None, cut_label=0):
    if corpus is None:
        return [], None

    vocab_label = get_vocab_label(corpus, vocab_label, cut_label)
    x = get_x(corpus, vocab_word)
    y = get_y(corpus, vocab_label)
    xy = concat_x_y(x, y)
    samples = get_samples(xy, init_emb)
    return samples, vocab_label


def main(argv):
    say('\nSYSTEM START')
    say('\nMODE: Training')
    say('\nRECURRENT HIDDEN UNIT: %s\n' % argv.unit)

    say('\tTRAINING\t\tBatch: %d  Epoch: %d  Parameters Save: %s' % (argv.batch, argv.epoch, argv.save))
    say('\tINITIAL EMBEDDING\t %s' % argv.init_emb)
    say('\tNETWORK STRUCTURE\tEmb Dim: %d  Hidden Dim: %d  Layers: %d' % (argv.emb, argv.hidden, argv.layer))
    say('\tOPTIMIZATION\t\tMethod: %s  Learning Rate: %f  L2 Reg: %f' % (argv.opt, argv.lr, argv.reg))

    ################
    # Load dataset #
    ################
    train_corpus, dev_corpus, test_corpus = get_dataset(argv)
    init_emb, vocab_word = get_init_emb(argv)

    ##################
    # Create samples #
    ##################
    # samples: 1D: n_samples, 2D: (x, y)
    say('\n\tCreating Training/Dev/Test Samples...')
    train_samples, vocab_label = create_samples(train_corpus, vocab_word, None, init_emb, argv.cut_label)
    dev_samples, vocab_label_dev = create_samples(dev_corpus, vocab_word, vocab_label, init_emb, 0)
    test_samples, vocab_label_test = create_samples(test_corpus, vocab_word, vocab_label, init_emb, 0)

    say('\tTrain Samples: %d' % len(train_samples))
    say('\tDev Samples: %d' % len(dev_samples))
    say('\tTest Samples: %d' % len(test_samples))
    say('\tLabel size: %d' % vocab_label.size())

    ##################
    # Create batches #
    ##################
    train_samples = get_batches(train_samples, argv.batch)
    say('\tTrain Batches: %d' % len(train_samples))

    ########################
    # dump arg/vocab dicts #
    ########################
    dump_data(data=vocab_label, fn='label.size-%d' % (vocab_label.size()))
    dump_data(data=vocab_word, fn='word.size-%d' % (vocab_word.size()))
    dump_data(data=init_emb, fn='emb.size-%d' % (len(init_emb)))

    #############
    # Model API #
    #############
    model_api = ModelAPI(argv, init_emb, vocab_word, vocab_label)
    model_api.set_model()
    model_api.set_train_f()
    model_api.set_test_f()

    ############
    # Training #
    ############
    say('\nTRAIN START')

    best_dev_f = 0.0
    f1_history = {}
    best_epoch = -1

    for epoch in xrange(argv.epoch):
        say('\nEpoch: %d' % (epoch + 1))
        model_api.train_all(train_samples)

        ###############
        # Development #
        ###############
        if dev_samples:
            say('  DEV')
            dev_f, predicts = model_api.predict_all(dev_samples, vocab_label_dev)

            if best_dev_f < dev_f:
                best_dev_f = dev_f
                best_epoch = epoch
                f1_history[best_epoch+1] = [best_dev_f]

                if argv.save:
                    fn = 'model.unit-%s.layer-%d.batch-%d.hidden-%d.reg-%f' %\
                         (argv.unit, argv.layer, argv.batch, argv.hidden, argv.reg)
                    dump_data(data=model_api.model, fn=fn)

                fn = 'result-dev.unit-%s.layer-%d.batch-%d.hidden-%d.opt-%s.reg-%f.txt' %\
                     (argv.unit, argv.layer, argv.batch, argv.hidden, argv.opt, argv.reg)
                save_predicted_prop(dev_corpus, vocab_label_dev, predicts, fn)

        ########
        # Test #
        ########
        if test_samples:
            say('  TEST')
            test_f, predicts = model_api.predict_all(test_samples, vocab_label_test)

            if epoch+1 in f1_history:
                f1_history[best_epoch+1].append(test_f)

#                output_results(test_corpus, te_prds, label_dict, predicts,
#                               'Test-result-%s.layer%d.batch%d.hidden%d.opt-%s.reg-%f.txt' % (
#                               argv.unit, argv.layer, argv.batch, argv.hidden, argv.opt, argv.reg))

        show_f1_history(f1_history)
