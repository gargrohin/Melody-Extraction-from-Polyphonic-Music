
'''
Code for CRNN model and it's training.
Also has the data_generator used. The neccessary reshaping of the matrices and normalization etc. done here.
Take note of the paths to the data_generator. 
Previous knowledge of generators and keras/tensorflow required to understand the code.
'''


import numpy as np
import librosa
import os

#import logging
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,Callback
import keras as k

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Reshape, BatchNormalization, Bidirectional, GRU,Dropout
from keras.layers import Conv2D, LSTM, Input, TimeDistributed, Lambda, ZeroPadding3D


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

import h5py
import json
import os
#import csv
import sys
#import pandas as pd
#import mir_eval
import math

from sklearn.preprocessing import LabelBinarizer,normalize

def train_model(model):
    '''
    The function that trains a certain neural network model with the given arguments.
    :param model: Keras.Model - Constructed model
    :param args: List - Input arguments
    :return:
    '''
    
# x_train, y_train, x_validation, y_validation = load_dataset_TD(dataset_number=args.dataset_number, args=args)
    #
    # dataset_train_size = x_train.shape[0]  # First dimension gives the number of samples
    # dataset_validation_size = x_validation.shape[0]

    batch_size = 16
    # Set the optimizers
    opt_ADAM = Adam(clipnorm=1., clipvalue=0.5)
    opt_SGD = SGD(lr=0.0005, decay=1e-4, momentum=0.9, nesterov=True)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=opt_ADAM, metrics=['accuracy'])

    # Use either a part of training set per epoch or all the set per epoch
    # if args.use_part_of_training_set_per_epoch:
    #     number_of_batches_train = np.int(np.floor(args.training_amount_number_of_samples/args.batch_size))
    # else:
    #     number_of_batches_train = np.max((np.floor((dataset_train_size) / args.batch_size), 1))
    #
    # number_of_batches_validation = np.max((np.floor(dataset_validation_size / args.batch_size), 1))

    # if args.use_part_of_training_set:
    #     filename = 'model{0}_' \
    #                'datasetNumber-{1}_' \
    #                'augment-{2}_patchSize-{3}_' \
    #                'numberOfPatches-{4}_' \
    #                'batchSize-{5}_' \
    #                'batchInOneEpoch-{6}_' \
    #                'trainingAmountPercentage-{7}'.format(
    #         args.model_name, args.dataset_number, args.augment_data, args.patch_size, args.number_of_patches,
    #         args.batch_size, number_of_batches_train, np.int(args.training_amount_percentage))
    # else:
    #     filename = 'model{0}_' \
    #                'datasetNumber-{1}_' \
    #                'augment-{2}_' \
    #                'patchSize-{3}_' \
    #                'numberOfPatches-{4}_' \
    #                'batchSize-{5}_' \
    #                'batchInOneEpoch-{6}'.format(
    #         args.model_name, args.dataset_number, args.augment_data, args.patch_size, args.number_of_patches,
    #         args.batch_size, number_of_batches_train)

    cb = set_callbacks()

    model.fit_generator(generator = generator(train_names),
                        steps_per_epoch = 85,
                        epochs = 100,
                        validation_data= generator(val_names),
                        validation_steps= 20,
                        callbacks= cb,
                        verbose= 1)

    #model.load_weights('{0}/{1}.h5'.format(get_trained_model_save_path(dataset_name=args.dataset_name), filename))

    return model

def sq(x):
    from keras import backend as K
    return K.squeeze(x, axis=4)


def construct_model():
    '''
    Construcs the CRNN model
    :param args: Input arguments
    :return: model: Constructed Model object
    '''
    number_of_patches = 20
    patch_size = 50
    feature_size = 301
    number_of_classes = 61
    step_notes = 5
    RNN = 'LSTM'
    verbose = False

    kernel_coeff = 0.00001

    number_of_channels = 1
    input_shape = (number_of_patches, patch_size, feature_size, number_of_channels)

    inputs = Input(shape=input_shape)


    zp = ZeroPadding3D(padding=(0, 0, 2))(inputs)

    #### CNN LAYERS ####
    cnn1 = TimeDistributed(Conv2D(64, (1, 5),
                                  padding='valid',
                                  activation='relu',
                                  strides=(1, np.int(step_notes)),
                                  kernel_regularizer=k.regularizers.l2(kernel_coeff),
                                  data_format='channels_last', name='cnn1'))(inputs)

    cnn1a = BatchNormalization()(cnn1)

    zp = ZeroPadding3D(padding=(0, 1, 2))(cnn1a)

    cnn2 = TimeDistributed(
        Conv2D(64, (3, 5), padding='valid', activation='relu', data_format='channels_last', name='cnn2'))(zp)

    cnn2a = BatchNormalization()(cnn2)

    zp = ZeroPadding3D(padding=(0, 1, 1))(cnn2a)

    cnn3 = TimeDistributed(
        Conv2D(64, (3, 3), padding='valid', activation='relu', data_format='channels_last', name='cnn3'))(zp)

    cnn3a = BatchNormalization()(cnn3)

    zp = ZeroPadding3D(padding=(0, 1, 7))(cnn3a)

    cnn4 = TimeDistributed(
        Conv2D(16, (3, 15), padding='valid', activation='relu', data_format='channels_last', name='cnn4'))(zp)

    cnn4a = BatchNormalization()(cnn4)

    cnn5 = TimeDistributed(
        Conv2D(1, (1, 1), padding='same', activation='relu', data_format='channels_last', name='cnn5'))(cnn4a)

    #### RESHAPING LAYERS ####
    cnn5a = Lambda(sq)(cnn5)

    cnn5b = Reshape((number_of_patches * patch_size, -1), name='cnn5-reshape')(cnn5a)

    #### BIDIRECTIONAL RNN LAYERS ####
    # if RNN == 'LSTM':
    #     rnn1 = Bidirectional(LSTM(128,
    #                               kernel_regularizer=k.regularizers.l1_l2(0.0001),
    #                               return_sequences=True), name='rnn1')(cnn5b)
    # elif RNN == 'GRU':
    #     rnn1 = Bidirectional(GRU(128,
    #                              kernel_regularizer=k.regularizers.l1_l2(0.0001),
    #                              return_sequences=True), name='rnn1')(cnn5b)

    #### CLASSIFICATION (DENSE) LAYER ####
    classifier = TimeDistributed(Dense(number_of_classes,
                                       activation='softmax',
                                       kernel_regularizer=k.regularizers.l2(0.00001),
                                       bias_regularizer=k.regularizers.l2()), name='output')(cnn5b)

    model = Model(inputs=inputs, outputs=classifier)

    if verbose == True or 1:
        model.summary()

        print('{0} as RNN!'.format(RNN))

    return model


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def song(sp, gp):

    HF0 = np.load(sp)
    HF0 = normalize(HF0)
    #HF0 = librosa.power_to_db(HF0, ref=np.max)

    #HF0 = HF0['arr_0']
    ground = np.load(gp)
    #print(gp)
    T = HF0.shape[1]
    # if T != ground.shape[0]:
    #     print(T, ground.shape[0])
    #     print('ground dimension error')

    gr = np.zeros([61,T])
    for t in range(T):
        if ground[t] != 0:
            gr[int(ground[t] -1) , t] = 1

    patches_size = 50
    number_of_patches = math.floor(T/patches_size)
    # print(number_of_patches)

    x = np.zeros([number_of_patches,patches_size,301])
    y = np.zeros([number_of_patches,patches_size,61])

    j = 0
    i=0
    while j < number_of_patches:
#         if i%patches_size == 0 and i !=0:
#             j+=1
#         if j +1 == number_of_patches:
#             break
        #if j == 10: print(i)
        x[j] = np.swapaxes(HF0[:,i:i+patches_size] , 0 , 1)
        y[j] = np.swapaxes(gr[:,i:i+patches_size] , 0,1)
        i = i + patches_size
        j+=1

    # print(y.shape)
    return x , y

names = np.load('names.npy')
l = names.shape[0]
train_names = names[:int(0.8*l)]
val_names = names[int(0.8*l):]

# def on_epoch_end(indices):
#         'Updates indexes after each epoch'
#         np.random.shuffle(indices)
#         return indices

def generator(names):
    sp = 'outs_hf/out_'
    gp = 'ground/'
    indices = np.arange(names.shape[0])
    while True:
        np.random.shuffle(indices)

        # if train:
        #     names = train_names
        # else:
        #     names = val_names
        count=0
        batch_size = 16
        patch_size = 50
        number_of_patches = 20
        x_train_batch = np.zeros([batch_size , number_of_patches , patch_size , 301 , 1 ])
        y_train_batch = np.zeros([batch_size , patch_size * number_of_patches , 61])
        batch_count = 0
        for i in indices:
            name = names[i]
            x, y = song(sp + name[:-1] + 'y', gp + name[:-1] + 'y')



            i = 0
            while (i+1)*number_of_patches <= x.shape[0]:
                count+=1


                x_train = np.reshape(
                            x[i*number_of_patches : (i+1)*number_of_patches,:,:] ,
                            [number_of_patches , patch_size , 301 , 1 ])
                y_train = np.zeros([number_of_patches * patch_size , 61])
                for j in range(number_of_patches):
                    y_train[j*patch_size : (j+1)*patch_size,  : ] = y[i+j]

                #print(y_train.shape)
                y_train = np.reshape(
                            y_train, [number_of_patches * patch_size , 61])

                i+=1

                if batch_count < batch_size:
                    x_train_batch[batch_count] = x_train
                    y_train_batch[batch_count] = y_train
                    batch_count +=1
                else:
                    batch_count = 0
                    yield x_train_batch, y_train_batch


        print()
        print(count)
        print()


def set_callbacks():
    '''
    Sets the callback functions for the network training

    :param save_filename: Filename to be used in ModelCheckpoint
    :param args: Input arguments
    :return: cb: List of callbacks
    '''

    # Callbacks
    cb = [EarlyStopping(monitor='val_loss',
                        patience=20,
                        verbose=True),
          ModelCheckpoint('model3.h5',
                          monitor='val_loss',
                          save_best_only=True,
                          verbose=False),
          ReduceLROnPlateau(monitor='val_loss',
                            patience=10,
                            verbose=True)]

    return cb





model = construct_model()
model_trained = train_model(model)
model_json = model_trained.to_json()
with open("model3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_trained.save_weights("model3.h5")
