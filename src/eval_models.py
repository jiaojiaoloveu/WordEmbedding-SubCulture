import os
from read_data import CorpusType
from gensim.models.word2vec import Word2Vec

if __name__ == '__main__':
    model_type = CorpusType.TWITTER
    model_name = 'word2vec_base_300'
    model_path = '../models/embedding/%s/%s' % (model_type.value, model_name)
    model = Word2Vec.load(model_path)
    result_path = '../result/%s/%s' % (model_type.value, model_name)
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
