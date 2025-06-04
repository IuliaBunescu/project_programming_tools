import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@st.spinner("Training model...")
def train_model_pipeline(numerical_cols, categorical_cols, model, task_type):
    """
    Trains a pipeline, separates a prediction row, and evaluates using the appropriate metric.

    Returns:
        pipeline     : trained sklearn pipeline
        eval_message : metric string
    """
    df = st.session_state.data.copy()

    prediction_row = df.iloc[[-1]].copy()
    st.session_state.prediction_row = prediction_row

    df = df.iloc[:-1].copy()

    # Transformers
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")
    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numerical_transformer, numerical_cols),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X = df[numerical_cols + categorical_cols]

    if task_type in ["Regression", "Classification"]:
        target_col = st.session_state.target[0]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        if task_type == "Regression":
            score = r2_score(y_test, y_pred)
            eval_message = f"R² Score: {score:.3f}"

        else:
            score = accuracy_score(y_test, y_pred)
            eval_message = f"Accuracy: {score:.3f}"

    elif task_type == "Clustering":
        pipeline.fit(X)
        cluster_labels = pipeline.named_steps["model"].labels_
        score = silhouette_score(X, cluster_labels)
        eval_message = f"Silhouette Score: {score:.3f}"

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return pipeline, eval_message
