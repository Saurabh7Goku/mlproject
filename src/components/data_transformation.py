import sys
from dataclasses import dataclass 
import numpy as np
import pandas as pd 
from src.utils import save_obj


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 

from src.exception import CustomException
from src.logger import logging 
import os
 
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts', 'preprocess_model.pkl') 

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        '''
        This function is responsible for the data transformation of different types of data
        eg : Strings, categorical, numerical etc. 
        '''
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = ["gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",]

            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                 ('scaler', StandardScaler())
                ]
            )

            logging.info('Done with numerical data scalling...')


            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoding', OneHotEncoder())
                ]
            )

            logging.info('Categorical data One hot encoded...')

            preprocessor = ColumnTransformer(
                [
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)  


    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('training and testing data loaded...')

            logging.info('obtaining preprocessing objects...')

            preprocessing_obj = self.get_data_transformation_object()

            target_column = 'math_score'
            #numerical_features = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns = [target_column], axis = 1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns = [target_column], axis = 1)
            target_feature_test_df = test_df[target_column]

            logging.info('Applying preprocessing objects on training and test dataframe...')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info('task 1 done')

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info('task 2 done')
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            

            logging.info('Saved Preprocessing object !!!')

            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys) 
        