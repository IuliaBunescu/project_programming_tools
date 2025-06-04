from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

ALGO_TYPES = ["Regression", "Classification", "Clustering"]

ML_ALGO_OPTIONS = {
    "Regression": XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
    ),
    "Classification": LogisticRegression(max_iter=1000),
    "Clustering": KMeans(),
}
