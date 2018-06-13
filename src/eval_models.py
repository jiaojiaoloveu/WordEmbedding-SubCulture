import os
import argparse
from corpus_type import decide_model_type
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="train models")
    ap.add_argument("--corpus", required=True, type=str, help="which corpus to use(github, wikitest, twitter)")
    ap.add_argument("--model", required=True, type=str, help="which model to use")
    args = vars(ap.parse_args())
    corpus_type = args.get("corpus")
    model_name = args.get("model")
    model_type = decide_model_type(model_name)

    model_path = '../models/embedding/%s/%s' % (corpus_type, model_name)
    if model_type == 'word2vec':
        model = Word2Vec.load(model_path)
    elif model_type == 'fasttext':
        model = FastText.load(model_path)
    else:
        model = []

    result_path = '../result/%s/%s' % (corpus_type, model_name)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    wordsim = model.wv.evaluate_word_pairs('../data/word353/wordsim353.tsv')
    question = model.wv.accuracy('../data/question/questions-words.txt')[-1]
    result_content = [
        model_path,
        str(model.wv.vectors.shape),
        str(wordsim),
        question,
        question['section'],
        len(question['correct']),
        len(question['incorrect']),
    ]
    with open(result_path, 'w+') as fp:
        fp.writelines("%s\n" % line for line in result_content)
