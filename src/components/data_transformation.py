import os,sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
import pickle

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocess_file_path=os.path.join('preprocess','preprocess.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            preprocess_obj=Pipeline(
                steps=[('impute',SimpleImputer(strategy='median')),
                      ('scale',StandardScaler())
                      ]
            )
            return preprocess_obj

        except Exception as e:
            logging.info('error occured in preprocess_obj')
            raise CustomException(sys,e)

    def initiate_data_transformation(self,train_file_path:str,test_file_path:str):
        try:
            logging.info('data transformation has started')

            train_df=pd.read_csv(train_file_path)
            test_df=pd.read_csv(test_file_path)

            logging.info('train data and test data read completed')
            Target_column='Concrete compressive strength(MPa, megapascals) '

            # train data
            input_fetures_train_df=train_df.drop(columns=[Target_column,'_id'],axis=1) # x_train
            target_features_train_df=train_df[Target_column] # y_train

            # test data
            input_fetures_test_df=test_df.drop(columns=[Target_column,'_id'],axis=1) #x_test
            target_features_test_df=test_df[Target_column] # y_test

            logging.info(f'input_fetures_train_df : {input_fetures_train_df.head().to_string()}')

            logging.info(f'input_fetures_test_df : {input_fetures_test_df.head().to_string()}')

            # preprocess data
            preprocesser=self.get_data_transformation_obj()

            

            transformed_input_train_feature=preprocesser.fit_transform(input_fetures_train_df)
            transformed_input_test_feature=preprocesser.transform(input_fetures_test_df)

            logging.info('preprocesser completed')

            # train_arr=np.c_[transform_input_features_test_df,np.array(target_features_train_df)]
            # test_arr=np.c_[transform_input_features_test_df,np.array(target_features_test_df)]
            train_arr = np.c_[transformed_input_train_feature, np.array(target_features_train_df) ]
            test_arr = np.c_[ transformed_input_test_feature, np.array(target_features_test_df) ]



            
           
            save_object(self.transformation_config.preprocess_file_path,
                        obj= preprocesser)
            logging.info("Exited initiate_data_transformation method of DataTransformation class")
            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocess_file_path
            )

    


        except Exception as e :
            logging.info('Error occured in data transformation')
            raise CustomException(sys,e)            


