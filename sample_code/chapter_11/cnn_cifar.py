#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cifar10
import functools

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])
  #params:
  dr_rate = 0.5 #rate at which units are dropped in training
  l2_reg = 0.0  #weight decay norm coefficient
  
  regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      kernel_regularizer=regularizer,
      activation=None)

  dropout1 = tf.layers.dropout(
      inputs=conv1, rate=dr_rate, training=mode == tf.estimator.ModeKeys.TRAIN)
  batch_norm1 = tf.layers.batch_normalization(inputs=dropout1, axis=3, 
          training=mode == tf.estimator.ModeKeys.TRAIN)
  nn1 = tf.nn.relu(batch_norm1)
  pool1 = tf.layers.max_pooling2d(inputs=nn1, pool_size=[2, 2], strides=2)
#  pool1 = tf.layers.max_pooling2d(inputs=dropout1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      kernel_regularizer=regularizer,
      activation=None)

  dropout2 = tf.layers.dropout(
      inputs=conv2, rate=dr_rate, training=mode == tf.estimator.ModeKeys.TRAIN)
  batch_norm2 = tf.layers.batch_normalization(inputs=dropout2, axis=3,
          training=mode == tf.estimator.ModeKeys.TRAIN)
  nn2 = tf.nn.relu(batch_norm2)  
  pool2 = tf.layers.max_pooling2d(inputs=nn2, pool_size=[2, 2], strides=2)
#  pool2 = tf.layers.max_pooling2d(inputs=dropout2, pool_size=[2, 2], strides=2)
  
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      kernel_regularizer=regularizer,
      activation=None)

  dropout3 = tf.layers.dropout(
      inputs=conv3, rate=dr_rate, training=mode == tf.estimator.ModeKeys.TRAIN)
  batch_norm3 = tf.layers.batch_normalization(inputs=dropout3, axis=3,
          training=mode == tf.estimator.ModeKeys.TRAIN)
  nn3 = tf.nn.relu(batch_norm3)  
  pool3 = tf.layers.max_pooling2d(inputs=nn3, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  #pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
  pool2_flat = tf.reshape(pool3, [-1, 4 * 4 * 128])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, 
          kernel_regularizer=regularizer, activation=None)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=dr_rate, training=mode == tf.estimator.ModeKeys.TRAIN)
  batch_norm = tf.layers.batch_normalization(inputs=dropout, axis=1,
          training=mode == tf.estimator.ModeKeys.TRAIN)
  nn_final = tf.nn.relu(batch_norm)  

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=nn_final, units=10)
#  logits = tf.layers.dense(inputs=dense, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  reg_loss = tf.losses.get_regularization_loss()
  loss = loss+reg_loss

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

    batch_accuracy, batch_accuracy_op = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]) 
    tf.summary.scalar("batch_accuracy", batch_accuracy)

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=tf.group(train_op, batch_accuracy_op))

  #eval_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "eval_loss": (loss, loss.op)}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
#mode=mode, loss=loss, eval_metric_ops=tf.group(eval_metric_ops, eval_loss.op))


def input_fn(data_dir,
             subset,
             batch_size):
  dataset = cifar10.Cifar10DataSet(data_dir, subset, subset=='train')
  image_batch, label_batch = dataset.make_batch(batch_size)
  #return [image_batch], [label_batch]

  return {"x": image_batch}, label_batch
def main(unused_argv):
  
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/cifar_convnet_model")

  train_input_fn = functools.partial(
        input_fn,
        data_dir="/mnt/data/cifar10",
        subset='train',
        batch_size=200)

  eval_input_fn = functools.partial(
        input_fn,
        data_dir="/mnt/data/cifar10",
        subset='eval',
        batch_size=10000)

  n_no_improve_local=0
  prev_loss_local = 1e100
  n_no_improve=0
  prev_loss = 1e100
  for i in range(0,300):
    train_results = mnist_classifier.train(
      input_fn=train_input_fn,
      steps=1000)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn, steps=1)
    if prev_loss > eval_results["eval_loss"]:
      n_no_improve=0
      prev_loss = eval_results["eval_loss"]
    else:
      n_no_improve=n_no_improve+1 
    
    if prev_loss_local > eval_results["eval_loss"]:
      n_no_improve_local=0
    else:
      n_no_improve_local=n_no_improve_local+1 
    prev_loss_local = eval_results["eval_loss"]

    if n_no_improve > 10 or n_no_improve_local > 5 :
      print("early stopping")
      break


  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn, steps=1)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
