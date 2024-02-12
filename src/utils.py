import os,sys
import pandas as pd 
import numpy as np
import pickle
import boto3
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from pymongo import MongoClient
from src.logger import logging
from src.exception import CustomException

def import_data_from_mongo(database_name, collection_name):
    # Connect to MongoDB
    try:
        client = MongoClient("mongodb+srv://www588650:arpan@cluster0.xyculx8.mongodb.net/?retryWrites=true&w=majority")  # Update with your MongoDB connection string

        # Select the database and collection
        db = client[database_name]
        collection = db[collection_name]

        # Query to retrieve data from the collection
        data = collection.find()

        df = pd.DataFrame(list(collection.find()))

    

        
        return df
    except Exception as e :
        logging.info('error in Mongo db')
        raise CustomException(sys,e)     


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path,obj):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)        

    except Exception as e:
        logging.info('error in lode_object')
        raise CustomException(sys,e)


def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(x_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(x_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report
    except Exception as e:
        logging.info('Error occured in evaluate model utils')
def download_model(bucket_name,bucket_file_name,dest_file_name):
    try:
        s3_client= boto3.client('s3')
        s3_client.download_file(bucket_name,bucket_file_name,dest_file_name)

        return dest_file_name

    except Exception as e:
        logging.info('error in downlod model')
        raise CustomException(e,sys)
     
      
def upload_file(from_filename,to_filename,bucket_name):
    try:
        s3_resouece=boto3.client('s3')
        s3_resouece.meta.client.upload(from_filename,to_filename,bucket_name)
    
    except Exception as e:
        logging.info('error in upload model')
        raise CustomException(e,sys)
      

        