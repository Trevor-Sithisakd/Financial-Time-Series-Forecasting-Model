from sklearn.linear_model import Ridge

MODEL_NAME = "LinearRegression (Ridge alpha=1)"


def build():
    return Ridge(alpha=1.0)
