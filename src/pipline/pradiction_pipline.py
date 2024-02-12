import os,sys,shutil
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object,upload_file,download_model

from flask import Flask,request

from dataclasses import dataclass

@dataclass
class PredictionFileDetail:
    prediction_output_dirname: str = "predictions"
    prediction_file_name:str =  "predicted_file.csv"
    prediction_file_path:str = os.path.join(prediction_output_dirname,prediction_file_name)



class PredictionPipeline:
    def __init__(self, request: request):

        self.request = request
        self.prediction_file_detail = PredictionFileDetail()


    def save_input_files(self)-> str:

        """
            Method Name :   save_input_files
            Description :   This method saves the input file to the prediction artifacts directory. 
            
            Output      :   input dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            logging.info('input file save process started')
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)
            logging.info('input file save dir')

            input_csv_file = self.request.files['file']
            logging.info('input file save done')

            pred_file_path = os.path.join(pred_file_input_dir,'input_file.csv')
            
            
            input_csv_file.save(pred_file_path)

            logging.info('exist from input files')

            return pred_file_path
        except Exception as e:
            logging.info('error in input file save ')

            raise CustomException(e,sys)

    def predict(self, features):
            try:
                logging.info('aws bucket dir')

                model_path = download_model(
                    bucket_name="ineuron-test-bucket-123",
                    bucket_file_name="model.pkl",
                    dest_file_name="model.pkl",
                )

                model = load_object(file_path=model_path)

                preds = model.predict(features)

                return preds

            except Exception as e:
                logging.info('eror in aws bucket dir')
                raise CustomException(e, sys)
        
    def get_predicted_dataframe(self, input_dataframe_path:pd.DataFrame):

        """
            Method Name :   get_predicted_dataframe
            Description :   this method returns the dataframw with a new column containing predictions

            
            Output      :   predicted dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
   
        try:
            logging.info('enter in  get_predicted_dataframe ')
            prediction_column_name : str = "class"
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)
            
            
            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in predictions]
            target_column_mapping = {0:'neg', 1:'pos'}

            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)
            
            os.makedirs( self.prediction_file_detail.prediction_output_dirname, exist_ok= True)
            input_dataframe.to_csv(self.prediction_file_detail.prediction_file_path, index= False)
            logging.info("predictions completed. ")



        except Exception as e:
            logging.info('error get_predicted_dataframe ')
            raise CustomException(e, sys) from e
        

        
    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_file_detail


        except Exception as e:
            logging.info('error run pipline ')
            raise CustomException(e,sys)
            