from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

x = tf.constant([[0,0], [0,1], [1,0], [1,1]], dtype=tf.float32)
y_true = tf.constant([[0], [1], [1], [0]], dtype=tf.float32)

#hidden_layer_model = tf.layers.Dense(units=2,activation=tf.nn.relu,use_bias=1)
#hidden_layer = hidden_layer_model(x)

hidden_layer = tf.layers.dense(x, units=2,activation=tf.nn.relu,use_bias=1,name="hidden_layer")

#y_pred = tf.layers.dense(hidden_layer,units=1,activation=tf.sigmoid,use_bias=1,name="output_layer")
y_pred = tf.layers.dense(hidden_layer,units=1,use_bias=1,name="output_layer")

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
#loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_true, logits=y_pred, reduction=tf.losses.Reduction.SUM)

optimizer = tf.train.GradientDescentOptimizer(0.005)
#optimizer = tf.train.AdagradOptimizer(0.001)
#optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

writer = tf.summary.FileWriter('./run_xor')
writer.add_graph(tf.get_default_graph())
tf.summary.scalar("loss", loss)

#weights, biases = hidden_layer_model.weights;
#tf.summary.histogram('weights', weights)
#tf.summary.histogram('biases', biases)

summary_op = tf.summary.merge_all()

sess.run(init)
for i in range(100000):
  _, loss_value, summary = sess.run((train, loss, summary_op))
  writer.add_summary(summary,i)
  #print(loss_value)
  if loss_value < 0.0001: 
    break

#output=tf.sigmoid(y_pred)
#print(sess.run(output))
print(sess.run(y_pred))
#print(sess.run(weights))
#print(sess.run(biases))
