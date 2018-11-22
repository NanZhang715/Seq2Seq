# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import data_helpers
import tflearn
import pandas as pd
import time
from sqlalchemy import create_engine



# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("sql", " select id, no, plat_name,send_time,send_phone,recv_phone,send_date,message_content from illegal_gather ", "SQL querys trainset  ")
tf.flags.DEFINE_string("stops_words", './stop_words.txt', "the file contains stop words")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_string('f', '', 'kernel')

FLAGS = tf.flags.FLAGS


# Set the logger
data_helpers.set_logger(os.path.join('./', 'train.log'))

    
## load data
x_raw, y_raw, data = data_helpers.load_data_and_labels_multiclass(FLAGS.sql,FLAGS.stops_words)

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
        all_predictions = np.empty([0,2])

        for x_test_batch in batches:
            batch_predictions, batch_scores = sess.run([predictions,scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])            
        
        pred = np.argmax(all_predictions, axis=1)
           
data['update_time']= time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))       
pred['result'] = pd.DataFrame(pred,columns=['result'])

df = pd.concat([data,pred], axis =1)

from sqlalchemy.dialects.mysql import INTEGER, VARCHAR

engine = create_engine("mysql+pymysql://root:Cncert@603@10.2.16.31:3306/funds_info_bd_stat_mongo?charset=utf8",encoding = 'utf-8')

dtypedict = {
        'id':INTEGER,
        'no':INTEGER,
        'plat_name':VARCHAR(255),
        'send_time':VARCHAR(255),
        'send_phone':VARCHAR(255),
        'recv_phone':VARCHAR(255),
        'send_date':VARCHAR(255),
        'message_content':VARCHAR(255),
        'update_time':VARCHAR(255),
        'result':VARCHAR(255)
            }

              
pd.io.sql.to_sql(df,
                 name='gathering',
                 con=engine,
                 schema= 'funds_info_bd_stat_mongo',
                 if_exists='append',
                 index= False,
                 dtype = dtypedict
                 )



