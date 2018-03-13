#test the custom estimator

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np

#import iris_data
# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
import autoenc_estimator as ae
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--train_steps', default=30000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    train_imgs = mnist.train.images
    train_expected = train_imgs

    test_imgs = mnist.test.images
    test_expected = test_imgs

    eval_imgs , _ = mnist.validation.next_batch(1)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": train_imgs},
   	y={"y": train_expected},
    	batch_size=args.batch_size,
    	num_epochs=None,
    	shuffle=True)
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": test_imgs},
   	y={"y": test_expected},
    	num_epochs=1,
    	shuffle=False)
    
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": eval_imgs},
   	y=None,
    	num_epochs=1,
    	shuffle=False)

    # Build 2 the autoencoder.
    classifier = tf.estimator.Estimator(
        model_fn=ae.my_model,
        model_dir="/tmp/mnist_ae",
        params={
            # Two hidden layers of 10 nodes each.
            'autoenc_units': [256, 128, 256],
        })

    # Train the Model.
    classifier.train(
        input_fn=train_input_fn,
        steps=args.train_steps)

    # Evaluate the model.
#    eval_result = classifier.evaluate(
#        input_fn=eval_input_fn)

#    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    predictions = classifier.predict(input_fn=predict_input_fn)

    gt = np.reshape(eval_imgs, [-1,28,28])
    #preds = np.reshape(predictions['reconstructed'], [-1, 28, 28])

    for pred_dict in predictions:
        preds = np.reshape(pred_dict['reconstructed'], [-1, 28, 28])
	print (preds.shape)
	plt.imshow(preds[0])
	plt.show()	
	plt.imshow(gt[0])
	plt.show()	


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
