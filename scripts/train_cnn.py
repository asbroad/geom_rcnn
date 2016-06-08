#!/usr/bin/env python

import rospy
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from load_data import load_data
import time
from IPython import embed
import cPickle
import Image
from matplotlib import cm, colors
import matplotlib.pyplot as plt

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

rospy.init_node('train_cnn', anonymous=True)

model_file = rospy.get_param('/train_cnn/model_file')
category_file = rospy.get_param('/train_cnn/category_file')
rootdir = rospy.get_param('/train_cnn/data_dir')
verbose = rospy.get_param('/train_cnn/verbose')

trX, teX, trY, teY, trIm, teIm, catagories = load_data(rootdir, 28, True)
inv_catagories = {v: k for k, v in catagories.items()}

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

X = T.ftensor4()
Y = T.fmatrix()

w = init_weights((32, 1, 3, 3))
w2 = init_weights((64, 32, 3, 3))
w3 = init_weights((128, 64, 3, 3))
w4 = init_weights((128 * 3 * 3, 625))
w_o = init_weights((625, len(catagories.keys())))

# save best model
temp_w = init_weights((32, 1, 3, 3))
temp_w2 = init_weights((64, 32, 3, 3))
temp_w3 = init_weights((128, 64, 3, 3))
temp_w4 = init_weights((128 * 3 * 3, 625))
temp_w_o = init_weights((625, len(catagories.keys())))
best_acc = 0

noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(2000):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    test_acc = np.mean(np.argmax(teY, axis=1) == predict(teX))
    print test_acc
    if test_acc > best_acc:
        best_acc = test_acc
        temp_w = w.get_value(borrow=True)
        temp_w2 = w2.get_value(borrow=True)
        temp_w3 = w3.get_value(borrow=True)
        temp_w4 = w4.get_value(borrow=True)
        temp_w_o = w_o.get_value(borrow=True)


# Save file
f = file(model_file, 'wb')
cPickle.dump(temp_w, f, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(temp_w2, f, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(temp_w3, f, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(temp_w4, f, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(temp_w_o, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
print('Saved Model to File : ' + str(model_file))

f = file(category_file, 'wb')
cPickle.dump(catagories, f, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(inv_catagories, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
print('Saved Category Set to File : ' + str(category_file))

# Print out image
predictions = predict(teX)

fig = plt.figure()
length = 10
for idx in range(length):#predictions.shape[0]):
    a=fig.add_subplot(1,length,idx)
    imgplot = plt.imshow(teIm[idx])
    plt.axis('off')
    a.set_title(inv_catagories[predictions[idx]])
plt.show()

# Calculate and create confusion matrix
conf_mat = np.zeros((len(catagories.keys()),len(catagories.keys())))
truth_data = np.argmax(teY,axis=1)
for idx in range(len(predictions)):
    conf_mat[predictions[idx]][truth_data[idx]] += 1

for idx1 in range(conf_mat.shape[0]):
    total = np.sum(conf_mat, axis=0)[idx1]
    for idx2 in range(conf_mat.shape[1]):
        conf_mat[idx1][idx2] = float(conf_mat[idx1][idx2]/total)

plt.matshow(conf_mat)
plt.show()
