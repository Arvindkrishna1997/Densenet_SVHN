from layers import *
from metrics import *
import numpy as np
import tensorflow as tf
import argparse
import os
import time

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

BATCH_SIZE = 64
flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        self.N = int(4)
        self.growthRate = ( (32 - 4)/4 )

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.layers2 = []
        self.activations2 = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    class MLP(Model):
        def __init__(self, placeholders, input_dim, **kwargs):
            super(MLP, self).__init__(**kwargs)

            self.inputs = placeholders['features']
            self.input_dim = input_dim
            self.labels = placeholders['labels']
            self.output_dim = 10
            self.placeholders = placeholders

            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

            self.build()

        def _loss(self):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.labels)
            cross_entropy = tf.reduce_mean(cross_entropy)*100
            self.loss += cross_entropy

        def _accuracy(self):
            correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        def _build(self):

            def conv(name, l, channel, stride):
                return Conv2D(name, l, channel, 3, stride=stride,
                              nl=tf.identity, use_bias=False,
                              W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))

            def add_layer(name, l, i):
                shape = l.get_shape().as_list()
                in_channel = shape[3]
                with tf.variable_scope(name) as scope:
                    c = BatchNorm('bn1_layers{}'.format(i), l)
                    c = tf.nn.relu(c)
                    c = conv('conv1_layers{}'.format(i), c, self.growthRate, 1)
                    l = tf.concat([c, l], 3)
                return l

            def add_transition(name, l, i):
                shape = l.get_shape().as_list()
                in_channel = shape[3]
                with tf.variable_scope(name) as scope:
                    l = BatchNorm('bn1_trans{}'.format(i), l)
                    l = tf.nn.relu(l)
                    l = Conv2D('conv1_trans{}'.format(i), l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
                    l = AvgPooling('pool', l, 2)
                return l


            l = conv('conv0', self.inputs, 16, 1)
            print("l input shape", l.get_shape())
            with tf.variable_scope('block1') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l, i)

            l = add_transition('transition1', l, i)

            with tf.variable_scope('block2') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l, i)

            l = add_transition('transition2', l, i)

            with tf.variable_scope('block3') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l, i)

            l = BatchNorm('bnlast', l)
            l = tf.nn.relu(l)
            l = GlobalAvgPooling('gap', l)
            self.outputs = FullyConnected('linear', l, out_dim=10, nl=tf.identity)



        def predict(self):
            return tf.nn.softmax(self.outputs)
