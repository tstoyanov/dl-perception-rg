#An estimator for the MNIST autoencoder

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_input = 784 # MNIST data input (img shape: 28*28)
def my_model(features, labels, mode, params):

    """Autoencoder on the MNIST data set"""
    #input layer
    input_layer = tf.reshape(features["x"], [-1, num_input])

    net = input_layer
    #Encoder
    for units in params['autoenc_units']:
        net = tf.layers.dense(net, units=units, activation=tf.sigmoid, use_bias=True)
    
    net = tf.layers.dense(net, units=num_input, use_bias=True, activation=tf.sigmoid) 
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'reconstructed': net
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
 
    # Compute loss.
    # expected_output = tf.reshape(labels["y"], [-1, num_input])
    loss = tf.reduce_mean(tf.pow(input_layer - net, 2))
 
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss)
 
    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
 
    #optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
