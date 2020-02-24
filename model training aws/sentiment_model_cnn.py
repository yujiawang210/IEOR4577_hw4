"""
Model definition for CNN sentiment training


"""

import os

import tensorflow as tf

from tensorflow import keras
# import keras.models 
# import Sequential
from tensorflow.keras import layers
from keras.layers import Dense, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

import boto3

def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling

    """

    cnn_model = tf.keras.Sequential()
    cnn_model.add(layers.Embedding(input_dim=config['embeddings_dictionary_size'],
                            output_dim=config['embeddings_vector_size'],
                            input_length=config['padding_size']))
    # cnn_model.add(layers.Embedding(input_dim=50000,
    #                     output_dim=1,
    #                     input_length=20))
    cnn_model.add(layers.Conv1D(filters=100,kernel_size=2,padding='valid',activation='relu',strides=1))
    cnn_model.add(layers.GlobalMaxPooling1D())
    cnn_model.add(layers.Dense(100, activation='relu'))
    # cnn_model.add(Activation('relu'))
    cnn_model.add(layers.Dense(1, activation = 'sigmoid'))
    # cnn_model.add(Activation('sigmoid'))
    cnn_model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    return cnn_model

def save_model(model):
    """
    Method to save models in SaveModel format with signature to allow for serving


    """

    # model.save(os.path.join(output))
    # model.save(output)

    # key = "{}/{}/examples".format(prefix,data_partition_name)
    # url = 's3://{}/{}'.format(bucket, key)
    # boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_file('data.csv')
    # print('Done writing to {}'.format(url))
    
    model.save('output/sentiment_model.h5')

    s3 = boto3.resource('s3')
    s3.meta.client.upload_file('output/sentiment_model.h5', 'ieor4577-hw4', 'sentiment_model.h5')

    # tf.saved_model.save(model, os.path.join(output, "1"))
    print("Model successfully saved")
