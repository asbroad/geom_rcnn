from __future__ import print_function
from sklearn.cross_validation import train_test_split
from PIL import Image
import numpy as np
import os

def load_data(datadir, sample_size, verbose):

    count = 0
    num_samples = get_num_samples(datadir)

    ys = np.zeros((num_samples,1), dtype=np.int)
    xs = np.zeros((num_samples,sample_size*sample_size), dtype=np.float)
    images = []
    catagories = {}
    catagory_count = 0

    for subdir, dirs, files in os.walk(datadir):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith('.jpg'):
                split_filepath = filepath.split('/')
                cat = split_filepath[-2]
                if cat not in catagories.keys():
                    catagories[cat] = catagory_count
                    catagory_count += 1
                ys[count] = catagories[cat]
                im = Image.open(filepath)
                images.append(im)
                im = im.resize((sample_size,sample_size)) # resize to 28 x 28
                im_gray = im.convert('L') # luma transform -  L = R * 299/1000 + G * 587/1000 + B * 114/1000
                im_gray_list = list(im_gray.getdata()) # make it into normal readable object
                im_gray_list_np = np.array(im_gray_list) / 255.0
                xs[count] = im_gray_list_np
                count += 1

    # Create Training / Testing Splits
    [xs_train, xs_test, ys_train, ys_test, ims_train, ims_test] = create_train_test_split(xs, ys, images, len(catagories.keys()))

    if verbose:
        print('Total Number of Samples : ', num_samples)
        print('Training Set Size       : ', xs_train.shape[0])
        print('Testing Set Size        : ', xs_test.shape[0])

    return [xs_train,xs_test,ys_train,ys_test,ims_train,ims_test,catagories]

def get_num_samples(datadir):

    sample_count = 0
    for subdir, dirs, files in os.walk(datadir):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith('.jpg'):
                sample_count += 1

    return sample_count

def one_hot(x,n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x),n))
    o_h[np.arange(len(x)),x] = 1
    return o_h

def create_train_test_split(xs, ys, images, num_catagories):
    train_test_split_idxs = np.array([itm for itm in range(0, xs.shape[0])])
    [idxs_train, idxs_test, temp1, temp2] = train_test_split(train_test_split_idxs,train_test_split_idxs,test_size=0.05, random_state=42) # test on 1/20th of the data?
    xs_train = xs[idxs_train]
    xs_test = xs[idxs_test]
    ys_train = ys[idxs_train]
    ys_test = ys[idxs_test]
    ims_train = [images[idx] for idx in idxs_train]
    ims_test = [images[idx] for idx in idxs_test]

    ys_train = one_hot(ys_train, num_catagories)
    ys_test = one_hot(ys_test, num_catagories)
    return [xs_train, xs_test, ys_train, ys_test, ims_train, ims_test]


if __name__=='__main__':
    [xs_train, xs_test, ys_train, ys_test, ims_train, ims_test, catagories] = load_data()
