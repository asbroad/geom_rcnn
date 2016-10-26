#!/usr/bin/env python

from __future__ import print_function

from sklearn.cross_validation import train_test_split
import cv2
import numpy as np
import os

def load_data(datadir, sample_size, train_test_split_percentage, verbose):

    # get number of data points in data directory
    num_samples = get_num_samples(datadir)
    # read in data
    xs, ys, catagories = get_data(datadir, num_samples, sample_size)
    # create training / testing splits
    xs_train, xs_test, ys_train, ys_test = create_train_test_split(xs, ys, len(catagories.keys()), train_test_split_percentage)

    if verbose: # print out information about dataset
        print('Total Number of Samples : ', num_samples)
        print('Training Set Size       : ', xs_train.shape[0])
        print('Testing Set Size        : ', xs_test.shape[0])

    return xs_train, xs_test, ys_train, ys_test, catagories

def get_data(datadir, num_samples, sample_size):
    xs = np.zeros((num_samples, 3, sample_size, sample_size), dtype=np.float)
    ys = np.zeros((num_samples,1), dtype=np.int)

    count = 0
    catagories = {}
    catagory_count = 0

    for subdir, dirs, files in os.walk(datadir):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith('_color.jpg'):
                split_filepath = filepath.split('/')
                cat = split_filepath[-2]
                if cat not in catagories.keys():
                    catagories[cat] = catagory_count
                    catagory_count += 1
                ys[count] = catagories[cat]

                im = cv2.resize(cv2.imread(filepath), (sample_size, sample_size)).astype(np.float32)
                im = im.transpose((2,0,1))
                im = np.expand_dims(im, axis=0)
                im = im.astype('float32')
                im = im/255.0

                xs[count] = im
                count += 1

    return xs, ys, catagories

def get_num_samples(datadir):
    sample_count = 0
    for subdir, dirs, files in os.walk(datadir):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith('_color.jpg'):
                sample_count += 1

    return sample_count

def one_hot(x,n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h

def create_train_test_split(xs, ys, num_catagories, train_test_split_percentage):
    train_test_split_idxs = np.array([itm for itm in range(0, xs.shape[0])])
    [idxs_train, idxs_test, temp1, temp2] = train_test_split(train_test_split_idxs, train_test_split_idxs, test_size=train_test_split_percentage, random_state=42)
    xs_train, xs_test = xs[idxs_train], xs[idxs_test]
    ys_train, ys_test = ys[idxs_train], ys[idxs_test]

    ys_train = one_hot(ys_train, num_catagories)
    ys_test = one_hot(ys_test, num_catagories)

    return [xs_train, xs_test, ys_train, ys_test]


if __name__=='__main__':
    [xs_train, xs_test, ys_train, ys_test, catagories] = load_data()
