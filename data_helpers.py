import jieba_fast.analyse as jiebanalyse
import codecs
import numpy as np
from functools import wraps
import time
import jieba_fast as jieba
from sqlalchemy import create_engine
import pandas as pd
import pymysql
import logging



def timeit(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        stime = time.clock()
        func(*args,**kwargs)
        endtime = time.clock()
        print("Runtime is {:.2f}s".format(endtime-stime))
        return func(*args,**kwargs)
    return wrapper


def load_data_and_labels_tagwords(positive_data_file, negative_data_file):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(codecs.open(positive_data_file, "r", "utf-8").readlines())
    positive_examples = [
        [item for item in jiebanalyse.extract_tags(s, withWeight=False, topK=20)] for
        s in positive_examples]
    negative_examples = list(codecs.open(negative_data_file, "r", "utf-8").readlines())
    negative_examples = [
        [item for item in jiebanalyse.extract_tags(s, withWeight=False, topK=20 )] for
        s in negative_examples]

    # Combine lists
    x_text = positive_examples + negative_examples

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]



def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(codecs.open(positive_data_file, "r", "utf-8").readlines())
    positive_examples = [
        [item for item in jieba.cut(s)] for s in positive_examples]
    negative_examples = list(codecs.open(negative_data_file, "r", "utf-8").readlines())
    negative_examples = [
        [item for item in jieba.cut(s)] for s in negative_examples]

    # Combine lists
    x_text = positive_examples + negative_examples

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    
    
    return [x_text, y]

#
#sql = '''
#    select id, corpus ,label from sms_model where  class= 'trainset'
#    '''
#stop_words ='./stop_words.txt'

def load_data_and_labels_multiclass(sql, stops_words):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
#    engine = create_engine("mysql+pymysql://root:Cncert@603@10.2.16.31:3306/funds_info_bd_stat_mongo?charset=utf8",encoding = 'utf-8')
    engine = create_engine("mysql+pymysql://root:Nathan715@127.0.0.1:3306/FlatWhite?charset=utf8",encoding = 'utf-8')
    
    # Load stop words
    with codecs.open(stops_words, "r", "utf-8") as file:
        stops_words = [line.strip() for line in file.readlines()]
    
    # Load data from files
    data = pd.read_sql(sql,con=engine) 
    x_corpus = data['corpus'].tolist()
    
    # Map the actual labels to one hot labels
    labels = sorted(set(data['label'].tolist()))
    one_hot = np.eye(len(labels),dtype = int)
    label_dict = dict(zip(labels, one_hot))
    
    x_raw = [[item for item in jieba.cut(s) if item not in stops_words] for s in x_corpus]
    y_raw = data['label'].map(lambda s : label_dict[s]).tolist()
    

    return x_raw, y_raw, data


def load_word2vector(filename):

    """
    Word2Vector_vocab – list of the words that we now have embeddings for
    Word2Vector_embed – list of lists containing the embedding vectors
    embedding_dict – dictionary where the words are the keys and the embeddings are the values
    """

    Word2Vector_vocab = []
    Word2Vector_embed = []
    embedding_dict = {}

    with codecs.open(filename, 'r',"utf-8") as file:
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab_word = row[0]
            Word2Vector_vocab.append(vocab_word)
            embed_vector = [float(i) for i in row[1:]]  # convert to list of float
            embedding_dict[vocab_word] = embed_vector
            Word2Vector_embed.append(embed_vector)

        print('Word2Vector Loaded Successfully')
        return Word2Vector_vocab, Word2Vector_embed, embedding_dict
    

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
 
           
@timeit
def load_data(data_file):
    """
    Loads data from files, extracts tag words
    """
    # Load data from files
    data = list(codecs.open(data_file, "r", "utf-8").readlines())
    x_text = [
        [item for item in jieba.extract_tags(s, withWeight=False, topK=20, allowPOS=())] for
        s in data]

    return x_text



def fetch_data_db(sql):

    ''' 
    IP/ICP Connect to the database  样本库

    '''

    connection = pymysql.connect(host='127.0.0.1',
                                 user='root',
                                 password='Nathan715',
                                 db='FlatWhite',
                                 charset='utf8mb4',
                                 port=3306,
                                 cursorclass=pymysql.cursors.DictCursor,
                                 connect_timeout=86400)

    try:
        with connection.cursor() as cursor:
            # Read a single record
            print(sql + ' is running')
            cursor.execute(sql)
            result = cursor.fetchall()
            result = pd.DataFrame(result)
            
    except ValueError:
        print("Error: Oopus, No Data, Drink a coffee")
        return None

    
    finally:
        connection.close()
        print(sql + 'is obtained')
        
    return result

