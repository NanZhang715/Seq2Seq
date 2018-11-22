#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 17:01:14 2018

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
from model.lstm_cnn import LSTM_CNN
from model.cnn_lstm import CNN_LSTM
from model.textcnn import TextCNN
from model.lstm import LSTM
import gensim
import logging
import utils


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("json_file", "", "json file containing parameters")

tf.flags.DEFINE_string("sql", " select id, corpus ,label from sms_model where  class= 'trainset' ", "SQL querys trainset")
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
    
utils.set_logger(os.path.join(log_folder, '{}_train.log'.format(params.model_version)))


def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    logging.info("Loading data...")
    x_text, y, _ = data_helpers.load_data_and_labels_multiclass(FLAGS.sql,
                                                             FLAGS.stops_words)

    sample_num = len(x_text)

    logging.info('{} samples in x_txt'.format(sample_num))
    logging.info('The min sentence contains ' + str(min([len(x) for x in x_text])))
    logging.info('The max sentence contains ' + str(max([len(x) for x in x_text])))

    # Build input array
    # - paddle to same length
    # - create dict and reverse dict with word ids
    
    max_document_length = max([len(x) for x in x_text])
    vocab_processor = tflearn.data_utils.VocabularyProcessor(max_document_length)
    text_list = []
    for text in x_text:
        text_list.append(' '.join(text))
    x = np.array(list(vocab_processor.fit_transform(text_list)))

    logging.info('The maximum of x is {}'.format(vocab_processor.max_document_length))
    logging.info('The number of vocab in train-set is {}'.format(len(vocab_processor.vocabulary_)))
    

    """
    Cautious:  embedding is not equal to reverse_dict !!
    
    """

    # Build Embedding array
    doc_vocab_size = len(vocab_processor.vocabulary_)

    # Extract word:id mapping from the object.
    vocab_dict = vocab_processor.vocabulary_._mapping

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

    logging.info('The shape of embedding is '.format(embedding.shape))

    # Randomly shuffle data
    x_shuffled, y_shuffled = tflearn.data_utils.shuffle(x, y)

    logging.info('The size of y_shuffled is {}'.format(len(y_shuffled)))
    logging.info('The size of x_shuffled is {}'.format(len(x_shuffled)))

    # Split train-set and test-set
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size = 0.2, random_state=0)


    logging.info('The size of train-set is {}'.format(len(x_train)))
    logging.info('The size of test-set is {}'.format(len(x_dev)))

    del x, y, x_shuffled, y_shuffled, embeddings_tmp

    return x_train, y_train, vocab_processor, x_dev, y_dev, embedding


def train(x_train, y_train, vocab_processor, x_dev, y_dev, embedding):
    
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        
        if params.model_version == 'TextCNN':
            model = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, params.filter_sizes.split(","))),
            num_filters=params.num_filters,
            l2_reg_lambda=params.l2_reg_lambda)
        
        elif params.model_version == 'CNN_LSTM':
            model = CNN_LSTM(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, params.filter_sizes.split(","))),
            num_hidden = params.num_hidden,
            num_filters=params.num_filters,
            l2_reg_lambda=params.l2_reg_lambda)        
            
        elif params.model_version == 'LSTM_CNN':
            model = LSTM_CNN(
            sequence_length = x_train.shape[1],
            num_classes = y_train.shape[1],
            vocab_size = len(vocab_processor.vocabulary_),
            embedding_size = FLAGS.embedding_dim,
            filter_sizes= list(map(int, params.filter_sizes.split(","))),
            num_filters=params.num_filters,
            l2_reg_lambda=params.l2_reg_lambda) 
                    
        elif params.model_version == 'LSTM':      
            model = LSTM(
            sequence_length = x_train.shape[1],
            num_classes = y_train.shape[1],
            vocab_size = len(vocab_processor.vocabulary_),
            num_hidden = params.num_hidden,
            embedding_size = params.embedding_dim,
            l2_reg_lambda=params.l2_reg_lambda
            )
            
        else:           
            raise AttributeError("No model found at model_dir") 

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
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob: params.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            logging.critical("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                model.input_x: x_batch,
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
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                logging.info("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev, embedding= preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev, embedding)


if __name__ == '__main__':
    tf.app.run()


