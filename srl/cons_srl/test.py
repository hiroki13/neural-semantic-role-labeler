from ..utils.io_utils import say
from ..utils.loader import load_conll, load_pos_tagged_corpus, load_data
from ..utils.saver import save_predicted_prop, save_predicted_srl, output_predicted_srl_to_cmd
from ..utils.preprocess import get_x, get_y, concat_x_y, get_samples, get_sample_x
from model_api import ModelAPI


def predict_pos_tagged_corpus(argv):
    ###############
    # Load corpus #
    ###############
    say('\n\tCorpus Preprocessing...')
    test_corpus = load_pos_tagged_corpus(argv.test_data, data_size=argv.data_size)
    say('\tTest  Sentences: %d' % (len(test_corpus) if test_corpus else 0))

    ##########################
    # Load initial embedding #
    ##########################
    say('\n\tInitial Embedding Loading...')
    init_emb = load_data(argv.load_emb)
    vocab_word = load_data(argv.load_word)
    vocab_label = load_data(argv.load_label)
    say('\tVocabulary Size: %d' % vocab_word.size())
    say('\tLabel size: %d' % vocab_label.size())

    ##################
    # Create samples #
    ##################
    # samples: 1D: n_samples, 2D: (x, y)
    x = get_x(test_corpus, vocab_word)
    test_samples = get_sample_x(x, init_emb)
    say('\tTest Samples: %d' % len(test_samples))

    #############
    # Model API #
    #############
    say('\nModel Loading...')
    model_api = ModelAPI(argv, init_emb, vocab_word, vocab_label)
    model_api.model = load_data(argv.load_model)
    model_api.set_pred_f()

    ########
    # Test #
    ########
    predicts = model_api.predict_all2(test_samples)

    if argv.output:
        output_predicted_srl_to_cmd(test_corpus, vocab_label, predicts)
    else:
        fn = 'result-srl.unit-%s.layer-%d.batch-%d.hidden-%d.opt-%s.reg-%f.txt' %\
             (argv.unit, argv.layer, argv.batch, argv.hidden, argv.opt, argv.reg)
        save_predicted_srl(test_corpus, vocab_label, predicts, fn)


def predict_conll_corpus(argv):
    ###############
    # Load corpus #
    ###############
    say('\n\tCorpus Preprocessing...')
    test_corpus = load_conll(argv.test_data, data_size=argv.data_size)
    say('\tTest  Sentences: %d' % (len(test_corpus) if test_corpus else 0))

    ##########################
    # Load initial embedding #
    ##########################
    say('\n\tInitial Embedding Loading...')
    init_emb = load_data(argv.load_emb)
    vocab_word = load_data(argv.load_word)
    vocab_label = load_data(argv.load_label)
    say('\tVocabulary Size: %d' % vocab_word.size())
    say('\tLabel size: %d' % vocab_label.size())

    ##################
    # Create samples #
    ##################
    # samples: 1D: n_samples, 2D: (x, y)
    x = get_x(test_corpus, vocab_word)
    y, vocab_label_test = get_y(test_corpus, vocab_label)
    xy = concat_x_y(x, y)
    test_samples = get_samples(xy, init_emb)
    say('\tTest Samples: %d' % len(test_samples))

    #############
    # Model API #
    #############
    say('\nModel Loading...')
    model_api = ModelAPI(argv, init_emb, vocab_word, vocab_label)
    model_api.model = load_data(argv.load_model)
    model_api.set_test_f()

    ########
    # Test #
    ########
    test_f, predicts = model_api.predict_all(test_samples, vocab_label_test)
    fn = 'result-test.unit-%s.layer-%d.batch-%d.hidden-%d.opt-%s.reg-%f.txt' %\
         (argv.unit, argv.layer, argv.batch, argv.hidden, argv.opt, argv.reg)
    save_predicted_prop(test_corpus, vocab_label_test, predicts, fn)


def main(argv):
    say('\nSYSTEM START')
    say('\nMODE: Test')
    say('\nRECURRENT HIDDEN UNIT: %s\n' % argv.unit)

    say('\tINITIAL EMBEDDING\t %s' % argv.init_emb)
    say('\tNETWORK STRUCTURE\tEmb Dim: %d  Hidden Dim: %d  Layers: %d' % (argv.emb, argv.hidden, argv.layer))

    if argv.data_type == 'conll':
        predict_conll_corpus(argv)
    else:
        predict_pos_tagged_corpus(argv)

