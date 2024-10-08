import sys 
import pandas as pd 
from src.exception import CustomException
from src.utils import load_object 

class PredictPipeline:
    def __init__(self):
        pass 

    def predict(self,features):

        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Check for missing values in the input features
            if features.isnull().sum().any():
                print("Missing values found in the input features:")
                print(features.isnull().sum())
                # Fill missing values with a placeholder (modify as needed)
                features = features.fillna({
                    'gender': 'unknown',  # Handle missing values for categorical variables
                    'race_ethnicity': 'unknown',
                    'parental_level_of_education': 'unknown',
                    'lunch': 'unknown',
                    'test_preparation_course': 'unknown'
            })

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds 
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        gender : str,
        race_ethnicity : str,
        parental_level_of_education : str,
        lunch : str,
        test_preparation_course : str,
        reading_score : int,
        writing_score : int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {

            "gender" :[self.gender],
            "race_ethnicity" : [self.race_ethnicity],
            "parental_level_of_education" :  [self.parental_level_of_education],
            "lunch" : [self.lunch], 
            "test_preparation_course" :  [self.test_preparation_course], 
            "reading_score" :  [self.reading_score], 
            "writing_score" :  [self.writing_score],
            
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)
        
