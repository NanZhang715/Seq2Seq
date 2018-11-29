#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:04:41 2018

@author: nzhang
"""

# -*- coding: utf-8 -*-

import tensorflow as tf


class Bi_LSTM(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, l2_reg_lambda=0.0,num_hidden=100):

        # PLACEHOLDERS
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")    # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")      # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")       # Dropout

        
        l2_loss = tf.constant(0.0) # Keeping track of l2 regularization loss

        #1. EMBEDDING LAYER ################################################################
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.constant(0.0, shape=[vocab_size, embedding_size]),
                trainable=True,
                name="W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            self.embedding_init = self.W.assign(self.embedding_placeholder)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)


        #2.Bi-LSTM LAYER ######################################################################
        with tf.name_scope("Bi-LSTM"):
            with tf.variable_scope('forward'):
                self.lstm_fw_cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
            #self.h_drop_exp = tf.expand_dims(self.h_drop,-1)
            with tf.variable_scope("backward"):
                self.lstm_bw_cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
                
            self.lstm_outputs, self.states = tf.nn.bidirectional_dynamic_rnn(cell_fw= self.lstm_fw_cell, 
                                                                             cell_bw= self.lstm_bw_cell, 
                                                                             inputs= self.embedded_chars,
                                                                             sequence_length=sequence_length, 
                                                                             dtype=tf.float32, 
                                                                             scope="BiLSTM")  
                
        #3. concat outpus and fetch the last output
        outputs = tf.concat(axis = 2, values = self.lstm_outputs) 
        last_output = outputs[:,-1,:]
        
        out_weight = tf.Variable(tf.random_normal([num_hidden, num_classes]))
        out_bias = tf.Variable(tf.random_normal([num_classes]))

        with tf.name_scope("output"):
            self.scores = tf.nn.xw_plus_b(last_output, out_weight,out_bias, name="scores")
            self.predictions = tf.nn.softmax(self.scores, name="predictions")

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses, name="loss")

        with tf.name_scope("accuracy"):
            self.correct_pred = tf.equal(tf.argmax(self.predictions, 1),tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"),name="accuracy")
            
        print('loaded Bi_LSTM')
