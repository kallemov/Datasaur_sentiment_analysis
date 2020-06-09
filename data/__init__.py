import pandas as pd
import numpy as np
import os 
from sklearn.model_selection import train_test_split

def create_training_data(opt):
    if opt.sentiment_analysis_type=='polarity':
        return create_polarity_data(opt)
    else:
        #not implemented for except for polarity
        return None
    
def create_polarity_data(opt):
    if opt.dataroot=='default':
        dataset_name='data/polarity3_data/twitter_polarity3.csv'
    else:
        dataset_name=opt.dataroot
    if os.path.exists(dataset_name):
        if dataset_name.lower().endswith(".json"):
            df=pd.read_json(dataset_name,orient='split')
            df=pd.concat([df,pd.DataFrame(df['sentence'].to_list())],axis=1).drop('sentence',axis=1)
        elif dataset_name.lower().endswith(".csv"):
            df = pd.read_csv(dataset_name, encoding='latin-1')
        else:
            print("%s file type is not supported." % dataset_name)
            exit(0)
        if opt.isTrain:
            label_dict = {'positive':0, 'negative':1, 'neutral':2}
            df['sentiment'] = df.sentiment.replace(label_dict)
            if df.shape[0]>opt.data_samples_max_size:
                df=df.sample(n=opt.data_samples_max_size,axis=0)
            X_train, X_val, y_train, y_val = train_test_split(df.text.values, 
                                                        df.sentiment.values, 
                                                        test_size=0.10, 
                                                        random_state=1, 
                                                        stratify=df.sentiment.values)
            return X_train, y_train, X_val, y_val
        else:
            return df.text.values
    else:
        print("%s path does not exist." % dataset_name)
        exit(0)
    
def create_predict_data(opt,returnDataFrame=False):
    dataset_name=opt.dataroot
    if os.path.exists(dataset_name):
        if dataset_name.lower().endswith(".json"):
            df=pd.read_json(dataset_name)
            df=pd.DataFrame(df['sentences'].to_list())
        elif dataset_name.lower().endswith(".csv"):
            df = pd.read_csv(dataset_name, encoding='latin-1')
        else:
            print("%s file type is not supported." % dataset_name)
            exit(0)
            
        if returnDataFrame:
            return df
        else:
            return df.text.values
    else:
        print("%s path does not exist." % dataset_name)
        exit(0)
    
