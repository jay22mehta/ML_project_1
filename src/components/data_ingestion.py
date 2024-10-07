import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass 
from src.components.data_transformation import DataTransformation,DataTransformationConfig



@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts','data.csv')
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')

        try:

            df = pd.read_csv('/Users/jaymehta/Desktop/ML_project_1/notebook/DATA/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw data saved')

            # Split the dataset into training and test sets
            logging.info('Train-test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data ingestion completed successfully')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)