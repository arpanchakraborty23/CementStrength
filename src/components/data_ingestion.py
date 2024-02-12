import os,sys
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import import_data_from_mongo
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_train import ModelTrainerConfig
from src.components.model_train import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")

    raw_data_path: str = os.path.join("artifacts", "data.csv")

    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info('Data ingestion has started')
        try:
            df: pd.DataFrame=import_data_from_mongo('PROJECT', 'CementStrength')

            # df=pd.read_csv('Notebook/data/cement_data.csv')
            # logging.info('data read completed')

            os.makedirs(
               os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path)

            train_data,test_data=train_test_split(df,test_size=0.25,random_state=20)

            train_data.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_data.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info(
                f"Ingested data from csv to {self.ingestion_config.raw_data_path}"
            )

            logging.info("Exited initiate_data_ingestion method of DataIngestion class")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )



        except Exception as e:
            logging.info('error occured in data ingestion')
            raise CustomException(sys,e)
if __name__=="__main__":
    obj=DataIngestion()
    tran_data,test_data=obj.initiate_data_ingestion() 

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(tran_data,test_data) 

    model_train=ModelTrainer()
    print(model_train.initate_model_training(train_arr,test_arr))

