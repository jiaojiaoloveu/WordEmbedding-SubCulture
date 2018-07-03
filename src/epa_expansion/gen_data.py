from propagate_labels import load_google_word_vectors, load_github_word_vectors
from align_wv_space import get_aligned_wv


def get_tokens():
    tokens = []
    # with open(os.path.join(word_dataset_base, 'wikitext-wordlist'), 'r') as fp:
    #     tokens = json.load(fp)
    # print('comparing %s' % len(tokens))
    tokens = ['good', 'nice', 'excellent', 'positive', 'warm', 'correct', 'superior',
              'bad', 'awful', 'nasty', 'negative', 'cold', 'wrong', 'inferior',
              'powerful', 'strong', 'potent', 'dominant', 'big', 'forceful', 'hard',
              'powerless', 'weak', 'impotent', 'small', 'incapable', 'hopeless', 'soft',
              'active', 'fast', 'noisy', 'lively', 'energetic', 'dynamic', 'quick', 'vital',
              'quiet', 'clam', 'inactive', 'slow', 'stagnant', 'inoperative', 'passive'
              ]
    return tokens


def wv_map():
    gg_model = load_google_word_vectors('../models/embedding/GoogleNews-vectors-negative300.bin')
    gh_model = load_github_word_vectors('../models/embedding/github/word2vec_sg_0_size_300_mincount_5')
    print('align wv space')
    tokens = get_tokens()
    dic = get_aligned_wv(gh_model.wv, gg_model, tokens)
    # gh_model, gg_model = align_models(gh_model, gg_model)
    # print('align done')
    # for w in get_tokens():
    #     if w in gg_model.vocab.keys() and w in gh_model.wv.vocab.keys():
    #         gg = gg_model[w]
    #         gh = gh_model.wv[w]
    #         dic[w] = (gg, gh)
    return dic

