import argparse
import os
import pickle

import MeCab

import preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='data')
    args = parser.parse_args()

    model_file = os.path.join(args.dir, 'model.pickle')
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    label_encoder_file = os.path.join(args.dir, 'label_encoder.pickle')
    with open(label_encoder_file, 'rb') as f:
        label_encoder = pickle.load(f)

    vectorizer_file = os.path.join(args.dir, 'vectorizer.pickle')
    with open(vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)

    tagger = MeCab.Tagger('-Owakati')

    while True:
        text = input()
        tokenized = preprocess.tokenize(tagger, text)

        x = vectorizer.transform([tokenized])
        y = model.predict(x)
        label = label_encoder.inverse_transform(y)[0]
        print('Tokenized:', tokenized)
        print('Label:', label)



if __name__ == '__main__':
    main()
