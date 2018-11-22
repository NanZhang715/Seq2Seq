# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import data_helpers
import tflearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import logging
import utils



# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("sql", " select id, corpus ,label from sms_model where  class= 'testset' ", "SQL querys trainset  ")
tf.flags.DEFINE_string("stops_words", './stop_words.txt', "the file contains stop words")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_string('f', '', 'kernel')

FLAGS = tf.flags.FLAGS

# Load the parameters from the experiment params.json file in params_dir
json_path = os.path.join(FLAGS.params_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params =utils.Params(json_path)

# create log folder
log_folder= './log'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Set the logger
utils.set_logger(os.path.join(log_folder, '{}_train.log'.format(params.model_version)))

    
## load data
x_raw, y_raw, data = data_helpers.load_data_and_labels_multiclass(FLAGS.sql,FLAGS.stops_words)
y_true = np.argmax(y_raw, axis=1)


# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
vocab_processor = tflearn.data_utils.VocabularyProcessor.restore(vocab_path)

text_list=[]
for text in x_raw:
    text_list.append(' '.join(text))
    
x_test = np.array(list(vocab_processor.transform(text_list)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir + 'checkpoints')
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]
        loss = graph.get_operation_by_name("output/scores").outputs[0]
        
        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), 
                                          FLAGS.batch_size, 
                                          1, 
                                          shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions, batch_scores = sess.run([predictions,scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])            
        
        pred = np.argmax(all_predictions, axis=1)
           
# Print accuracy if y_test is defined
if y_true is not None:
    correct_predictions = float(sum(all_predictions == y_raw))
    logging.critical("Total number of test examples: {}".format(len(y_raw)))
    logging.critical("Accuracy: {:g}".format(correct_predictions/float(len(y_raw))))

  
y_pred = list(pred)

logging.critical(classification_report(y_true, y_pred))
logging.critical(confusion_matrix(y_true, y_pred))



