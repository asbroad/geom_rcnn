#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
from PIL import Image as PImage
from cv_bridge import CvBridge, CvBridgeError
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import numpy as np
import cPickle

class CNN:

    def __init__(self, model_file, category_file, verbose):
        self.bridge = CvBridge()
        self.model_file = model_file
        self.category_file = category_file
        self.sample_size = 28
        self.srng = RandomStreams()

        f = file(self.category_file, 'rb')
        self.catagories = cPickle.load(f)
        self.inv_catagories = cPickle.load(f)
        f.close()

        X = T.ftensor4()
        Y = T.fmatrix()

        w = self.init_weights((32, 1, 3, 3))
        w2 = self.init_weights((64, 32, 3, 3))
        w3 = self.init_weights((128, 64, 3, 3))
        w4 = self.init_weights((128 * 3 * 3, 625))
        self.w_o = self.init_weights((625, len(self.catagories.keys())))

        noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = self.model(X, w, w2, w3, w4, 0.2, 0.5)
        l1, l2, l3, l4, py_x = self.model(X, w, w2, w3, w4, 0., 0.)
        y_x = py_x

        cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
        params = [w, w2, w3, w4, self.w_o]
        updates = self.RMSprop(cost, params, lr=0.001)

        train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        self.predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

        f = file(self.model_file, 'rb')
        w.set_value(cPickle.load(f), borrow=True)
        w2.set_value(cPickle.load(f), borrow=True)
        w3.set_value(cPickle.load(f), borrow=True)
        w4.set_value(cPickle.load(f), borrow=True)
        self.w_o.set_value(cPickle.load(f), borrow=True)
        f.close()

    def floatX(self, X):
        return np.asarray(X, dtype=theano.config.floatX)

    def init_weights(self, shape):
        return theano.shared(self.floatX(np.random.randn(*shape) * 0.01))

    def rectify(self, X):
        return T.maximum(X, 0.)

    def softmax(self, X):
        e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
        return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

    def dropout(self, X, p=0.):
        if p > 0:
            retain_prob = 1 - p
            X *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            X /= retain_prob
        return X

    def RMSprop(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
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

    def model(self, X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
        ''' This model can be replaced with any the author would like,
        however, it must match the model that is used to train the
        network.  This model is inspired by one developed by Alec Radford
        at indico.io '''
        l1a = self.rectify(conv2d(X, w, border_mode='full'))
        l1 = max_pool_2d(l1a, (2, 2))
        l1 = self.dropout(l1, p_drop_conv)

        l2a = self.rectify(conv2d(l1, w2))
        l2 = max_pool_2d(l2a, (2, 2))
        l2 = self.dropout(l2, p_drop_conv)

        l3a = self.rectify(conv2d(l2, w3))
        l3b = max_pool_2d(l3a, (2, 2))
        l3 = T.flatten(l3b, outdim=2)
        l3 = self.dropout(l3, p_drop_conv)

        l4 = self.rectify(T.dot(l3, w4))
        l4 = self.dropout(l4, p_drop_hidden)

        pyx = self.softmax(T.dot(l4, self.w_o))
        return l1, l2, l3, l4, pyx


def main():
    cnn = CNN()
    rospy.init_node('cnn_node', anonymous=True)
    try:
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")

if __name__=='__main__':
    main()
