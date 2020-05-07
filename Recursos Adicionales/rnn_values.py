#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:26:00 2020

@author: juangabriel
"""

## Importar las librerías
import numpy as np
import tensorflow as tf
from collections import namedtuple

# Setting the Hyperparameters

MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3
MODE_ZH = 4
HyperParams = namedtuple('HyperParams', ['num_steps',
                                         'max_seq_len',
                                         'input_seq_width',
                                         'output_seq_width',
                                         'rnn_size',
                                         'batch_size',
                                         'grad_clip',
                                         'num_mixture',
                                         'learning_rate',
                                         'decay_rate',
                                         'min_learning_rate',
                                         'use_layer_norm',
                                         'use_recurrent_dropout',
                                         'recurrent_dropout_prob',
                                         'use_input_dropout',
                                         'input_dropout_prob',
                                         'use_output_dropout',
                                         'output_dropout_prob',
                                         'is_training',
                                        ])

# Making a function that returns all the default hyperparameters of the MDN-RNN model

def default_hps():
  return HyperParams(num_steps=2000,
                     max_seq_len=1000,
                     input_seq_width=32+3,
                     output_seq_width=32,
                     rnn_size=256,
                     batch_size=100,
                     grad_clip=1.0,
                     num_mixture=5,
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=0,
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)



# Getting and sampling these default hyperparameters

hps = default_hps()


# Construcción de la RNN
num_mixture = hps.num_mixture
KMIX = num_mixture
INWIDTH = hps.input_seq_width
OUTWIDTH = hps.output_seq_width
LENGTH = hps.max_seq_len
if hps.is_training:
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell
use_recurrent_dropout = False if hps.use_recurrent_dropout == 0 else True
use_input_dropout = False if hps.use_input_dropout == 0 else True
use_output_dropout = False if hps.use_output_dropout == 0 else True   
use_layer_norm = False if hps.use_layer_norm == 0 else True
if use_recurrent_dropout:
    cell = cell_fn(hps.rnn_size, layer_norm = use_layer_norm, 
                   dropout_keep_prob = hps.recurrent_dropout_prob)
else:
    cell = cell_fn(hps.rnn_size, layer_norm = use_layer_norm)
if use_input_dropout:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob = hps.input_dropout_prob)
if use_output_dropout:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = hps.output_dropout_prob)
cell = cell
sequence_lengths = LENGTH
input_x = tf.placeholder(dtype = tf.float32, 
                              shape = [hps.batch_size, sequence_lengths, INWIDTH])
output_x = tf.placeholder(dtype = tf.float32, 
                               shape = [hps.batch_size, sequence_lengths, OUTWIDTH])
actual_input_x = input_x
initial_state = cell.zero_state(batch_size = hps.batch_size, dtype = tf.float32)
NOUT = OUTWIDTH * KMIX * 3
with tf.variable_scope("RNN"):
    output_w = tf.get_variable("output_w", shape = [hps.rnn_size, NOUT])
    output_b = tf.get_variable("output_b", shape = [NOUT])
output, last_state = tf.nn.dynamic_rnn(cell,
                                        actual_input_x,
                                        initial_state=initial_state,
                                        time_major=False,
                                        swap_memory=True,
                                        dtype=tf.float32,
                                        scope="RNN")
 