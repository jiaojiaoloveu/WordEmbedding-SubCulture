import os
import argparse
from utils import load_model
import json


def main():
    for method in ['word2vec', 'fasttext']:
        for sg in ['0', '1']:
            for size in range(200, 401, 50):
                for mincount in range(0, 21, 5):
                    model_name = '%s_sg_%s_size_%s_mincount_%s' % (method, sg, size, mincount)
                    model_path = '../models/embedding/%s/%s' % (corpus_type, model_name)
                    print(model_path)
                    model = load_model(model_path=model_path)
                    result_path = '../result/%s/%s' % (corpus_type, model_name)
                    os.makedirs(os.path.dirname(result_path), exist_ok=True)

                    pearson, spearman, oov_ratio = model.wv.evaluate_word_pairs('../data/word353/wordsim353.tsv')
                    question = model.wv.accuracy('../data/question/questions-words.txt')[-1]
                    result_content = [
                        pearson,
                        spearman,
                        oov_ratio,
                        len(question['correct']),
                        len(question['incorrect']),
                    ]
                    with open(result_path, 'w+') as fp:
                        json.dump(result_content, fp)
                    del model


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="train models")
    ap.add_argument("--corpus", required=True, type=str, help="which corpus to use(github, wikitest, twitter)")
    # ap.add_argument("--model", required=True, type=str, help="which model to use")
    args = vars(ap.parse_args())
    corpus_type = args.get("corpus")
    # model_name = args.get("model")

    main()

    # model_path = '../models/embedding/%s/%s' % (corpus_type, model_name)
    # model = load_model(model_path=model_path)

    # result_path = '../result/%s/%s' % (corpus_type, model_name)
    # os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # wordsim = model.wv.evaluate_word_pairs('../data/word353/wordsim353.tsv')
    # question = model.wv.accuracy('../data/question/questions-words.txt')[-1]
    # result_content = [
    #     model_path,
    #     str(model.wv.vectors.shape),
    #     str(wordsim),
    #     question,
    #     question['section'],
    #     len(question['correct']),
    #     len(question['incorrect']),
    # ]
    # with open(result_path, 'w+') as fp:
    #     fp.writelines("%s\n" % line for line in result_content)
