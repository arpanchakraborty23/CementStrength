import os,sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_train import ModelTrainer
from src.logger import logging
from src.exception import CustomException

class TrainPipline:
    def __init__(self):
        self.data_ingestion=DataIngestion()
        self.data_transformation=DataTransformation()
        self.model_train=ModelTrainer()

    def run_pipline(self):
        logging.info('run  pippline has started')
        try:
            train_path,test_path=self.data_ingestion.initiate_data_ingestion()
            (
                train_arr,test_arr,preprocess_file)=self.data_transformation.initiate_data_transformation(
                    train_file_path=train_path,test_file_path=test_path,preprocess_file=preprocess_file_path
                )
            
            r2_score=self.model_train.initate_model_training(
                train_array=train_arr,test_array=test_arr,
            )
            print("training completed. Trained model score : ", r2_square)

        except Exception as e:
                logging.info('error in train pipline')
                raise CustomException(e, sys) 
