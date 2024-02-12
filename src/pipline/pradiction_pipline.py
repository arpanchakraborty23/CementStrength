import os,sys,shutil
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object,upload_file,download_model

from flask import Flask,request

from dataclasses import dataclass

@dataclass
class PradictionPiplineConfig:
    pradiction_file_path=os.path.join('pradiction','pradiction.csv')


class PradictionPipline:
    def __init__(self,request:request):
        self.request= request
        self.pradiction_config=PradictionPiplineConfig()

    def save_input_files(self):

        """
            Method Name :   save_input_files
            Description :   This method saves the input file to the prediction artifacts directory. 
            
            Output      :   input dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
        """
        try:
            # input file location
            input_pradiction_file='pred_artifacts'
            os.makedirs(input_pradiction_file,exist_ok=True)

            input_csv_file=self.request.files['file']
            pred_file_path=os.path.join(input_pradiction_file,input_csv_file)

            input_csv_file.save(pred_file_path)
            
            return pred_file_path

        except Exception as e:
            logging.info('error in save input file')
            raise CustomException(e,sys)

    def pradict(self,features):
        try:
            model_path=download_model(
                bucket_name='ineuron-test-bucket-123',
                bucket_file_name='model.pkl',
                dest_file_name='model.pkl'
            )
            model= load_object(file_path=model_path)

            preds=model.pradict(features)

        except Exception as e:
            raise CustomException(e, sys)

    def pradicted_df(self,input_dataFrame_path:pd.DataFrame):
        """
            Method Name :   predicted_dataframe
            Description :   this method returns the dataframw with a new column containing predictions

            
            Output      :   predicted dataframe
            On Failure  :   Write an exception log and then raise an exception
        """ 
        try:
            pradiction_col_name :str ='class'

            input_dataFrame: pd.DataFrame=pd.read_csv(input_dataFrame_path)
            pradictions=self.predict(input_dataFrame)
            input_dataFrame[pradiction_col_name]=[pred for pred in pradictions]

            target_col_mapping={0:'neg',1:'pos'}

            input_dataFrame[pradiction_col_name]=input_dataFrame[pradiction_col_name].map(target_col_mapping)

            os.makedirs(self.pradiction_config.pradiction_output_dir,exist_ok=True)
            input_dataFrame.to_csv(self.pradiction_config.pradiction_file_path,index=False)


            logging.info("predictions completed. ")



        except Exception as e:
            raise CustomException(e, sys) from e
            

        def run_pipeline(self):
            try:
                input_csv_path=self.save_input_files()
                pradiction_result=self.pradicted_df(input_csv_path)

                return pradiction_result

                



            except Exception as e:
                logging.info('error in run pipline')
                raise CustomException(e, sys) from e        

    




