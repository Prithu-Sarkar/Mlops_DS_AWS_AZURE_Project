import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info("Loading model and preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info("Artifacts loaded successfully.")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(str(e), sys.exc_info())


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(str(e), sys.exc_info())


# ------------------------
# Test block
# ------------------------
if __name__ == "__main__":
    try:
        logging.info("Prediction pipeline test started.")

        # Create a sample student
        sample_student = CustomData(
            gender="female",
            race_ethnicity="group B",
            parental_level_of_education="bachelor's degree",
            lunch="standard",
            test_preparation_course="none",
            reading_score=78,
            writing_score=72
        )

        # Convert to DataFrame
        input_df = sample_student.get_data_as_data_frame()

        # Make prediction
        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)

        logging.info(f"Predicted math_score: {prediction[0]}")

    except Exception as e:
        logging.error(f"Prediction pipeline failed: {e}")
