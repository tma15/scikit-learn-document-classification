import argparse
import pickle
import os

from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='data')
    args = parser.parse_args()

    train_file = os.path.join(args.dir, 'train.pickle')
    with open(train_file, 'rb') as f:
        train_X, train_y = pickle.load(f)
        print('train', train_X.shape, train_y.shape)

    test_file = os.path.join(args.dir, 'test.pickle')
    with open(test_file, 'rb') as f:
        test_X, test_y = pickle.load(f)
        print('test:', test_X.shape, test_y.shape)

    model = Perceptron(
        penalty='l2',
        shuffle=True,
        verbose=2)

    model.fit(train_X, train_y)
    test_y_pred = model.predict(test_X)

    precision, recall, fscore, _ = precision_recall_fscore_support(
        test_y,
        test_y_pred,
        average='micro')

    print('Precision:', precision)
    print('Recall:', recall)
    print('F-score:', fscore)

    model_file = os.path.join(args.dir, 'model.pickle')
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()
