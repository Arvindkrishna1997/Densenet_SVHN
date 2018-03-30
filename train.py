from __future__ import division
from __future__ import print_function

import glob
import random
import base64
import pandas as pd

from PIL import Image
from io import BytesIO
from IPython.display import HTML

import time
import tensorflow as tf

from utils import *
from models import MLP
import numpy as np
import pandas as pd

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import h5py
from sklearn.preprocessing import OneHotEncoder

hf = h5py.File('SVHN_greyscale.h5', 'r')
hf.keys()
x_train = hf['x_train']
y_train = hf['y_train']
x_val = hf['x_val']
y_val = hf['y_val']



x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
hf.close()

X = tf.placeholder(tf.float32, [None, 32, 32, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

def plot_images(img, labels, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat):
        print("i", i ," ax ", ax)
        print(img[i].shape)
        im = img[i]
        if img[i].shape == (32, 32, 3):
            plt.subplot(ax)
            plt.imshow(img[i])
            print("vada")
        else:
            plt.subplot(ax)
            plt.gray()
            print("gray")
            plt.imshow(img[i,:,:,0])
        plt.subplots_adjust(wspace=0.2, hspace=0)
        ax.set_title(labels[i])
        ax.set_xticks([]);ax.set_yticks([])
    plt.show()

#plot_images(x_train,y_train, 2,8)
def dense_to_one_hot(labels_dense):
    enc = OneHotEncoder().fit(labels_dense.reshape(-1, 1))
    labels_one_hot = enc.transform(labels_dense.reshape(-1,1)).toarray()
    return labels_one_hot

y_train = dense_to_one_hot(y_train)
y_train = y_train.astype(np.uint8)
y_val = dense_to_one_hot(y_val)
y_val = y_val.astype(np.uint8)

# Split data into training & validation
validation_images = (x_val)
validation_labels = (y_val)

train_images = x_train
train_labels = y_train


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'dense', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 800, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1_1', 100, 'Number of units in hidden layer 1 of first branch')
flags.DEFINE_integer('hidden1_2', 100, 'Number of units in hidden layer 1 of second branch')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

features = X

#print(X.get_shape().as_list())

num_supports = 1
model_func = "MLP"

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'labels_mask': tf.placeholder(tf.int32),
    'features': tf.placeholder(tf.float32, [None, 32, 32, 1]),
    'labels': tf.placeholder(tf.float32, [None, 10]),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # 11helper variable for sparse dropout
}

# Create model
with TowerContext('', is_training=True):
    model = MLP(placeholders, input_dim=784, logging=True)


#Helper functions
epochs_completed = 0
index_in_epoch = 0
num_examples = x_train.shape[0]

# serve data by batches
def next_batch(batch_size):

    global x_train
    global y_train
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        x_train = x_train[perm]
        y_train = y_train[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return x_train[start:end], y_train[start:end]

# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []
# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(features, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
with TowerContext('', is_training=True):
    sess.run(tf.global_variables_initializer())

cost_val = []
VALIDATION_SIZE=4000
# Train model
#print(validation_images[0:100].shape," shape da")

#print(validation_images[0:100])
summary_writer = tf.summary.FileWriter('./output_values/logs', graph=sess.graph)
for epoch in range(FLAGS.epochs):

    batch_X, batch_Y = next_batch(100)


    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(batch_X, batch_Y, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    perm = np.arange(100)
    np.random.shuffle(perm)
    # Validation
    cost, acc, duration = evaluate(validation_images[perm], validation_labels[perm], placeholders)
    cost_val.append(cost)
    tf.summary.scalar("train_accuracy", (outs[2]))
    tf.summary.scalar("validation_acccuracy",(acc))

    merged_summary_op = tf.summary.merge_all()
    #print("outs",(outs[2]),"acc",(acc))
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    summary_str = sess.run(merged_summary_op, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, epoch)

    validation_accuracies.append(acc)
    train_accuracies.append(outs[2])
    x_range.append(epoch+1)

print("Optimization Finished!")
#print(train_accuracies)
plt.plot(x_range, train_accuracies,'-b', label='Training_data')
plt.plot(x_range, validation_accuracies,'-g', label='Validation_data')
plt.legend(loc='lower right', frameon=False)
plt.ylim(ymax = 1.0, ymin = 0)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.show()
