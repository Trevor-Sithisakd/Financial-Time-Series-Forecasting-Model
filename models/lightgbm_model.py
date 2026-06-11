from lightgbm import LGBMRegressor

MODEL_NAME = "LightGBM"

def build():
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        verbose=-1,
    )