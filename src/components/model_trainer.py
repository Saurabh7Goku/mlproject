import os
import sys
from dataclasses import dataclass
from src.utils import save_obj, evaluate_models 
from src.exception import CustomException 
from src.logger import logging

# Algorithms imports
from catboost import CatBoostRegressor 
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor 
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor 
from xgboost import XGBRegressor 
from sklearn.metrics import r2_score 

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting the training and test data')
            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),

            }

            model_report : dict = evaluate_models(
                X_train =X_train, y_train = y_train, 
                X_test = X_test, y_test = y_test, 
                models = models
                )

            best_model_score = max(sorted(model_report.values()))

            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model is found !!!")
            
            logging.info("Best found Model on both training and testing dataset...")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            y_pred = best_model.predict(X_test)

            r2_square = r2_score(y_test, y_pred)
            
            return r2_square
        

        except Exception as e:
            raise CustomException(e, sys)

