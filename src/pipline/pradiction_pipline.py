import os,sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object,upload_file,download_model

from flask import request

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
            pred_file_path=os.path.join(pred_file_path,input_csv_file)

            input_csv_file.save(pred_file_path)
            
            return pred_file_path

        except Exception as e:
            logging.info('error in save input file')
            raise CustomException(e,sys)

    def pradict(self,features):
        try:
            model_path=download_model(
                bucket_name='AC',
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
            pradiction_col_name='class'
            input_dataFrame=pd.DataFrame(input_dataFrame_path)
            pradictions=self.pradict(pradiction_col_name)
            input_dataFrame[pradiction_col_name]=[preds for pred in pradictions]

            target_col_mapping={0:'neg',1:'pos'}

            input_dataFrame[pradiction_col_name]=input_dataFrame[pradiction_col_name].map(target_col_mapping)

            os.makedirs(self.pradiction_config.pradiction_output_dir,exist_ok=True)
            input_dataFrame.to_csv(self.pradiction_config.pradiction_file_path,index=False)


            logging.info("predictions completed. ")



        except Exception as e:
            raise CustomException(e, sys) from e
            

        def run_pipline(self):
            try:
                input_csv_path=self.save_input_files()
                self.pradicted_df(input_csv_path)

                return self.pradiction_config

                



            except Exception as e:
                logging.info('error in run pipline')
                raise CustomException(e, sys) from e        

    




