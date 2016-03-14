#!/usr/bin/env python3                                                                                    # -*- coding:utf-8 -*-    

'''
This program estimates accuracy of sift-bof classifier.
Usage : make bof_hist of dataset.
python accuracy_test_bof.py NAME_OF_TEST_DATASET NAME_OF_CLASSIFIER_TO_TEST
'''
import argparse
import sys
import gzip
import cPickle as pickle
from sklearn.metrics import classification_report, accuracy_score


def accuracy_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',
        help='dataset must have data, target, target_names attributes. bof_hist.pkl.gz')
    parser.add_argument('classifier')
    args = parser.parse_args(sys.argv[1:])

    print('loading dataset')
    with gzip.open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)
    print('loading classifier')
    with gzip.open(args.classifier, 'rb') as f:
        clf = pickle.load(f)

    X_test = dataset.data
    y_test = dataset.target
    target_names = dataset.target_names
    y_pred = clf.predict(X_test)
    print('score of classifier: {}'.format(accuracy_score(y_test, y_pred))) 
    print(classification_report(y_test, y_pred, target_names=target_names))


def main():
    accuracy_test()

if __name__ == "__main__":
    main()
