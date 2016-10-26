#!/usr/bin/env python

from keras.models import Model
from keras.models import model_from_json
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, History

from load_data import load_data

import numpy as np
import pickle
import rospy
import time

from matplotlib import ticker
import matplotlib.pyplot as plt

class CNN:

    def __init__(self, data_dir, model_architecture, model_filename, weights_filename, categories_filename, history_filename, train_test_split_percentage, num_training_epochs, verbose):
        self.sample_size = 224 # this needs to match the input size of the pre-trained network, or you need to remove the input layer from the current network and replace it with the input size you would like
        self.optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
        self.train_test_split_percentage = train_test_split_percentage
        self.data_dir = data_dir
        self.model_architecture = model_architecture
        self.model_filename = model_filename
        self.weights_filename = weights_filename
        self.categories_filename = categories_filename
        self.history_filename = history_filename
        self.num_training_epochs = num_training_epochs
        self.verbose = verbose

    def load_dataset(self):
        if self.verbose:
            print 'loading data ... '
            start_time = time.time()

        self.xs_train, self.xs_test, self.ys_train, self.ys_test, self.categories = load_data(self.data_dir, self.sample_size, self.train_test_split_percentage, self.verbose)
        self.inv_categories = {v: k for k, v in self.categories.items()}

        num_val = len(self.xs_train)/10
        self.xs_val = self.xs_train[-num_val:]
        self.ys_val = self.ys_train[-num_val:]
        self.xs_train = self.xs_train[:-num_val]
        self.ys_train = self.ys_train[:-num_val]

        if self.verbose:
            end_time = time.time()
            self.print_time(start_time,end_time,'loading data')

    def make_model(self):
        # create the base pre-trained model
        if self.model_architecture == 'vgg16':
            from keras.applications.vgg16 import VGG16
            self.base_model = VGG16(weights='imagenet', include_top=False)
        elif self.model_architecture == 'resnet':
            from keras.applications.resnet50 import ResNet50
            self.base_model = ResNet50(weights='imagenet', include_top=False)
        elif self.model_architecture == 'inception':
            from keras.applications.inception_v3 import InceptionV3
            self.base_model = InceptionV3(weights='imagenet', include_top=False)
        else:
            print 'Model architecture parameter unknown. Options are: vgg16, resnet, and inception'
            rospy.signal_shutdown("Model architecture unknown.")

        # now we add a new dense layer to the end of the network inplace of the old layers
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        # add the outplut layer
        predictions = Dense(len(self.categories.keys()), activation='softmax')(x)

        # create new model composed of pre-trained network and new final layers
        # if you want to change the input size, you can do this with the input parameter below
        self.model = Model(input=self.base_model.input, output=predictions)

        # now we go through and freeze all of the layers that were pretrained
        for layer in self.base_model.layers:
            layer.trainable = False

        if self.verbose:
            print 'compiling model ... '
            start_time = time.time()

        # in finetuning, these parameters can matter a lot, it is wise to observe 
        # how well your model is learning for this to work well
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        if self.verbose:
            end_time = time.time()
            self.print_time(start_time,end_time,'compiling model')

    def data_augmentation(self):
        if self.verbose:
            print 'enhancing training set with data augmentation... '
            start_time = time.time()

        self.datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            fill_mode='nearest',
            horizontal_flip=True,
            vertical_flip=True
            )
        self.datagen.fit(self.xs_train)

        if self.verbose:
            end_time = time.time()
            self.print_time(start_time,end_time,'data augmentation')

    def finetune_model(self):

        if self.verbose:
            print 'training model ... '
            start_time = time.time()

        self.checkpointer = ModelCheckpoint(filepath=self.weights_filename, verbose=1, save_best_only=True)
        self.history = History()

        self.model.fit_generator(self.datagen.flow(self.xs_train, self.ys_train, batch_size=32),
                    samples_per_epoch=len(self.xs_train), nb_epoch=self.num_training_epochs,
                    validation_data=(self.xs_val, self.ys_val),
                    callbacks=[self.checkpointer, self.history])

        if self.verbose:
            end_time = time.time()
            self.print_time(start_time, end_time,'training model')

    def save_model(self):
        # we don't save model weights here, the best performing one has 
        # already been saved during the training
        json_string = self.model.to_json()
        open(self.model_filename, 'w').write(json_string) # save model architecture as JSON
        with open(self.categories_filename, 'wb') as outfile:
            pickle.dump(self.categories, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.history_filename, 'wb') as outfile:
            pickle.dump(self.history.history, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        self.model = model_from_json(open(self.model_filename).read())
        self.model.load_weights(self.weights_filename)
        self.categories = np.load(self.categories_filename)
        with open(self.categories_filename, 'r') as infile:
            self.categories = pickle.load(infile)
        self.inv_categories = {v: k for k, v in self.categories.items()}

        if self.verbose:
            print 'compiling model ... '
            start_time = time.time()

        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        if self.verbose:
            end_time = time.time()
            self.print_time(start_time,end_time,'compiling model')

    def test_model(self):
        if self.verbose:
            print 'testing model'
            start_time = time.time()

        predictions = self.model.predict(self.xs_test)
        self.predictions_int = np.argmax(predictions,1)
        self.true_ys_int = np.argmax(self.ys_test,1)
        num_preds_correct = np.sum(self.predictions_int == self.true_ys_int)
        total_test_cases = self.predictions_int.shape[0]
        model_accuracy = num_preds_correct/float(total_test_cases)

        if self.verbose:
            end_time = time.time()
            self.print_time(start_time, end_time, 'testing model')

        print '-------------------------------------------------'
        print 'Tested model on ' + str(total_test_cases) + ' images'
        print 'Number of corrrect predictions : ' + str(num_preds_correct)
        print 'Accuracy of model : ' + str(model_accuracy)
        print '-------------------------------------------------'

    def plot_test_examples(self):
        # Print example test images and their labels
        fig = plt.figure()
        length = 10
        for idx in range(length):
            a=fig.add_subplot(1,length,idx+1)
            im = self.xs_test[idx].transpose((1,2,0))
            plt.imshow(im)
            plt.axis('off')
            a.set_title(self.inv_categories[self.predictions_int[idx]])
        plt.show()

    def plot_confusion_matrix(self):
        # Calculate and create confusion matrix
        conf_mat = np.zeros((len(self.categories.keys()),len(self.categories.keys())))
        for idx in range(len(self.predictions_int)):
            conf_mat[self.predictions_int[idx]][self.true_ys_int[idx]] += 1

        for idx1 in range(conf_mat.shape[0]):
            total = np.sum(conf_mat, axis=0)[idx1]
            for idx2 in range(conf_mat.shape[1]):
                conf_mat[idx1][idx2] = float(conf_mat[idx1][idx2]/total)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        cax = ax.matshow(conf_mat)
        fig.colorbar(cax)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_xticklabels([''] + self.inv_categories.values(), rotation='vertical')
        ax.set_yticklabels([''] + self.inv_categories.values())

        plt.show()

    def print_time(self, start, end, function_name):
        print function_name + ' took ' + str(end-start) + ' seconds.'

def main():
    # set up parmaeters
    data_dir = rospy.get_param('/keras_cnn_finetune/data_dir')
    model_architecture = rospy.get_param('/keras_cnn_finetune/model_architecture')
    model_filename = rospy.get_param('/keras_cnn_finetune/model_file')
    weights_filename = rospy.get_param('/keras_cnn_finetune/weights_file')
    categories_filename = rospy.get_param('/keras_cnn_finetune/category_file')
    history_filename = rospy.get_param('/keras_cnn_finetune/history_file')
    verbose = rospy.get_param('/keras_cnn_finetune/verbose')
    train_test_split_percentage = rospy.get_param('/keras_cnn_finetune/train_test_split_percentage')
    num_training_epochs = rospy.get_param('/keras_cnn_finetune/num_training_epochs')

    # make model class
    cnn = CNN(data_dir, model_architecture, model_filename, weights_filename, categories_filename, history_filename, train_test_split_percentage, num_training_epochs, verbose)

    cnn.load_dataset()
    cnn.make_model()
    cnn.data_augmentation()
    cnn.finetune_model() # this also saves the weights (the model weights that perform best on the validation split)
    cnn.save_model() # this saves the model structure, object categories, and training history
    cnn.load_model() # now we need to load the best model
    cnn.test_model()
    cnn.plot_test_examples()
    cnn.plot_confusion_matrix()

    # If you're interested in viewing information related to the training history,
    # you can access the stored information in these variables
    # cnn.history.history['loss']
    # cnn.history.history['acc']
    # If you want to view the model architecture, you can fnd it in 
    # cnn.model

if __name__=='__main__':
    rospy.init_node('keras_cnn_finetune', anonymous=True)
    main()
    try:
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")
