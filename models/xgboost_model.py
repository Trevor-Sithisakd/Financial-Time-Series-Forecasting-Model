from xgboost import XGBRegressor
from config import MODEL_PARAMS

MODEL_NAME = "XGBoost"


def build():
    return XGBRegressor(**MODEL_PARAMS)
