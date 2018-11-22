#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:37:15 2018

@author: nzhang
"""

import pandas as pd   
import re
from sqlalchemy import create_engine
import data_helper

if __name__ == '__main__':

        sql = '''select id,
                        message_content,
                        label
                from corpus ;
        '''
        sms= data_helper.fetch_data_db(sql)       
        sms.drop_duplicates(keep = 'first',inplace =True)        
        sms=sms.fillna('')
                
        sms['corpus'] = sms['message_content'].map(lambda s: re.sub(r'[^\u4E00-\u9FA5]','',str(s)))

        print(sms.columns)
        
        sms = sms[['id',
                   'corpus',
                   'label']]
        
        y = sms['label']
        X = sms.iloc[:,0:3]
        
        print(X.columns)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size = 0.2, 
                                                            random_state = 0)
        
        trainset = pd.concat([X_train,y_train], axis = 1)
        testset = pd.concat([X_test,y_test], axis=1)
        
        
        trainset['class'] = 'trainset'
        testset['class']= 'testset'
        
        dataset = pd.concat([trainset, testset], axis = 0, ignore_index = True)
        dataset.reset_index(inplace = True, drop =True)
        
        print(dataset.columns)
        
#        engine = create_engine("mysql+pymysql://root:Cncert@603@10.2.16.31:3306/funds_info_bd_stat_mongo?charset=utf8",encoding = 'utf-8')     
        engine = create_engine("mysql+pymysql://root:Nathan715@127.0.0.1:3306/FlatWhite?charset=utf8",encoding = 'utf-8')

        from sqlalchemy.dialects.mysql import TEXT, INTEGER, LONGTEXT
        
        dtypedict = {
                'id':INTEGER,
                'corpus':LONGTEXT,
                'label':INTEGER,
                'class':TEXT}
        
                      
        pd.io.sql.to_sql(dataset,
                         name='sms_model',
                         con=engine,
                         schema= 'FlatWhite',
                         if_exists='append',
                         index= False,
                         dtype = dtypedict
                         )
        
        #  'id', 'corpus', 'label', 'label', 'class'
        print(dataset['class'].value_counts())
        