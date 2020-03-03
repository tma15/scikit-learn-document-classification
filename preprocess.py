import argparse
import csv
import os
import pickle

import pandas as pd
import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def read_file(dir_name):
    titles = []
    for file_name in os.listdir(dir_name):
        path = os.path.join(dir_name, file_name)
        with open(path) as f:
            _ = next(f)  # url
            _ = next(f)  # title
            title = next(f)
            titles.append(title.strip())
    return titles


def read_data(dir_name, tagger):
    categories = []
    titles = []
    for dir_or_file in os.listdir(dir_name):
        path = os.path.join(dir_name, dir_or_file)
        if os.path.isdir(path):
            titles_of_category = read_file(path)
            categories += [dir_or_file for _ in range(len(titles_of_category))]
            titles += [tokenize(tagger, t) for t in titles_of_category]

    df = pd.DataFrame({'category': categories, 'title': titles})
    print(df)
    return df


def tokenize(tagger, text):
    result = tagger.parse(text).strip()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    parser.add_argument('--out', default='data')
    args = parser.parse_args()

    tagger = MeCab.Tagger('-Owakati')

    df = read_data(args.dir, tagger)

    train, test = train_test_split(df)
    print(len(train), len(test))

    vectorizer = TfidfVectorizer(
        input='content')
    train_titles = train['title']
    train_X = vectorizer.fit_transform(train_titles)

    test_titles = test['title']
    test_X = vectorizer.transform(test_titles)

    label_encoder = LabelEncoder()
    train_categories = train['category']
    train_y = label_encoder.fit_transform(train_categories)

    test_categories = test['category']
    test_y = label_encoder.transform(test_categories)

    print(train_X.shape, train_y.shape)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    train_file = os.path.join(args.out, 'train.pickle')
    test_file = os.path.join(args.out, 'test.pickle')
    vectorizer_file = os.path.join(args.out, 'vectorizer.pickle')
    label_encoder_file = os.path.join(args.out, 'label_encoder.pickle')

    with open(train_file, 'wb') as f:
        pickle.dump([train_X, train_y], f)

    with open(test_file, 'wb') as f:
        pickle.dump([test_X, test_y], f)

    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)

    with open(label_encoder_file, 'wb') as f:
        pickle.dump(label_encoder, f)


if __name__ == '__main__':
    main()
