from sklearn.ensemble import RandomForestRegressor

MODEL_NAME = "RandomForest"

def build():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
    )