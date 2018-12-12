#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:35:02 2018

@author: nzhang
"""


import tensorflow as tf
import numpy as np
import tflearn
from sklearn.model_selection import train_test_split
import os
import time
import datetime
import data_helpers
from model.Parallel_CNN_LSTM import Parallel_CNN_LSTM
import gensim
import logging
import utils

import warnings
warnings.filterwarnings("ignore")

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("json_file", "Parallel_CNN_LSTM.json", "json file containing parameters")

tf.flags.DEFINE_string("sql", " select title, keywords, description, corpus ,label from p2p_model where  class= 'testset' ", "SQL querys trainset")
tf.flags.DEFINE_string("params_dir", "./params", "Directory containing params.json")

tf.flags.DEFINE_string("stops_words", './stop_words.txt', "the file contains stop words")
tf.flags.DEFINE_string("pre_trained_Word2vector", "/Users/nzhang/P2P/embed/head.txt", "Data source for Word2vector.")
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# Load the parameters from the experiment params.json file in params_dir
json_path = os.path.join(FLAGS.params_dir, FLAGS.json_file)
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params =utils.Params(json_path)

print(params.dict)


# create log folder
log_folder= './log'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

time_str = datetime.datetime.now().isoformat()   
utils.set_logger(os.path.join(log_folder, '{}_train_{}.log'.format(params.model_version,time_str)))


def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    logging.info("Loading data...")
    x_text, x_meta, y, _ = data_helpers.load_data_and_labels_multiclass(FLAGS.sql,
                                                                          FLAGS.stops_words,
                                                                          topn=20)
    sample_num_text = len(x_text)
    sample_num_meta = len(x_meta)
    
    max_length_text = max([len(x) for x in x_text])
    min_length_text = min([len(x) for x in x_text])
    logging.info('{} samples in x_txt'.format(sample_num_text))
    logging.info('The min sentence in body contains ' + str(min_length_text))
    logging.info('The max sentence in body contains ' + str(max_length_text))
    
    max_length_meta = max([len(x) for x in x_meta])
    min_length_meta = min([len(x) for x in x_meta])
    logging.info('{} samples in x_meta'.format(sample_num_meta))
    logging.info('The min sentence in meta contains ' + str(min_length_meta))
    logging.info('The max sentence in meta contains ' + str(max_length_meta))
    
    for item in x_text:
        item.extend(['<UNK>']*(max_length_text-len(item)))
        
    
    x_hstack = []
    for s in range(len(x_text)):
        x = x_text[s] + x_meta[s]
        x_hstack.append(x)
        
    logging.info('The length of x_hstack is {}'.format(len(x_hstack))) 
    
    # Build input array
    # - paddle to same length
    # - create dict and reverse dict with word ids
    
    max_document_length = max([len(x) for x in x_hstack])
    vocab_processor = tflearn.data_utils.VocabularyProcessor(max_document_length)
    text_list = []
    for text in x_text:
        text_list.append(' '.join(text))
    x = np.array(list(vocab_processor.fit_transform(text_list)))

    logging.info('The maximum of x is {}'.format(vocab_processor.max_document_length))
    logging.info('The number of vocab in train-set is {}'.format(len(vocab_processor.vocabulary_)))
    
    # Extract word:id mapping from the object.
    vocab_dict = vocab_processor.vocabulary_._mapping
    
    x = np.where(x==vocab_dict['UNK'], 0, x)
    
    """
    Cautious:  embedding is not equal to reverse_dict !!
    
    """

    # Build Embedding array
    doc_vocab_size = len(vocab_processor.vocabulary_)

    # Sort the vocabulary dictionary on the basis of values(id).
    # Both statements perform same task.
    # sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
    dict_as_list = sorted(vocab_dict.items(), key=lambda x: x[1])
    
    # load pre-trained Word2Vector
    model = gensim.models.KeyedVectors.load_word2vec_format(FLAGS.pre_trained_Word2vector,
                                                            binary= False)

    embeddings_tmp = []

    for i in range(doc_vocab_size):
        item = dict_as_list[i][0]
        if item in model.index2word:
            embeddings_tmp.append(model.get_vector(item))
        else:
            rand_num = np.random.uniform(low=-0.2, high=0.2, size=FLAGS.embedding_dim)
            embeddings_tmp.append(rand_num)

    # final embedding array corresponds to dictionary of words in the document
    embedding = np.asarray(embeddings_tmp)

    logging.info('The shape of embedding is {}'.format(embedding.shape))

    # Randomly shuffle data
    x_shuffled, y_shuffled = tflearn.data_utils.shuffle(x, y)

    logging.info('The size of y_shuffled is {}'.format(len(y_shuffled)))
    logging.info('The size of x_shuffled is {}'.format(len(x_shuffled)))

    # Split train-set and test-set
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size = 0.2, random_state=0)


    logging.info('The size of train-set is {}'.format(len(x_train)))
    logging.info('The size of test-set is {}'.format(len(x_dev)))

    del x, y, x_shuffled, y_shuffled, embeddings_tmp
    
    return x_train, y_train, vocab_processor, x_dev, y_dev, embedding, max_length_text, max_length_meta


def train(x_train, y_train, vocab_processor, x_dev, y_dev, embedding, max_length_text, max_length_meta):
    
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        
        if params.model_version == 'Parallel_CNN_LSTM':  
            model = Parallel_CNN_LSTM(sequence_length_b = max_length_text, 
                                      sequence_length_m = max_length_meta, 
                                      num_classes = y_train.shape[1], 
                                      vocab_size = len(vocab_processor.vocabulary_),
                                      filter_sizes = list(map(int, params.filter_sizes.split(","))),
                                      embedding_size=FLAGS.embedding_dim,
                                      num_filters = params.num_filters, 
                                      l2_reg_lambda=params.l2_reg_lambda,
                                      num_hidden=params.num_hidden)
           
        else:           
            raise AttributeError("No model found at model_dir folder") 

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp =  '{}_'.format(params.model_version) + str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        logging.critical("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=params.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        init = tf.global_variables_initializer()

        sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: embedding})
        sess.run(init) 
        
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                model.input_x_b: tuple([s[:max_length_text] for s in list(x_batch)]),
                model.input_x_m: tuple([s[max_length_text:] for s in list(x_batch)]),
                model.input_y: y_batch,
                model.dropout_keep_prob: params.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict)
            logging.critical("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                model.input_x_b: tuple([s[:max_length_text] for s in list(x_batch)]),
                model.input_x_m: tuple([s[max_length_text:] for s in list(x_batch)]),
                model.input_y: y_batch,
                model.dropout_keep_prob: params.dropout_keep_prob
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            logging.critical("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), params.batch_size, params.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % params.evaluate_every == 0:
                logging.info("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                logging.info("")
            if current_step % params.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                logging.info("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev, embedding, max_length_text, max_length_meta = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev, embedding, max_length_text, max_length_meta)


if __name__ == '__main__':
    tf.app.run()
