# Simple Streamlit ML App

This is a modular and interactive Streamlit app for exploring, training, and using machine learning models. The app is designed for simplicity and flexibility, supporting regression, classification, and clustering workflows.

---

## Features

- User authentication with both Google and Microsoft
- Upload and preview CSV datasets
- Automatic detection of numerical and categorical columns
- Null value detection and removal
- Select target and feature columns
- Train models for:
  - Regression (XGBoost)
  - Classification (Logistic Regression)
  - Clustering (KMeans)
- Evaluate model performance with appropriate metrics
- Make predictions using saved model
---

## Example Datasets

Use the example CSV files in the `data/` directory:

- `tips.csv`: Useful for binary classification (e.g. `smoker` column as target)
- `iris.csv`: Useful for clustering (e.g. `species` as ground truth for visualization)

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/streamlit-ml-app.git
   cd streamlit-ml-app
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the app:
   ```bash
   streamlit run app.py
   ```

---

## Dependencies

- streamlit
- pandas
- scikit-learn
- xgboost
- plotly
- joblib

Install all with:
```bash
pip install -r requirements.txt
```

---

## Notes

- The trained model is saved as `./model/model_pipeline.pkl`
- Missing values in prediction inputs will disable prediction until filled
- Uses `ColumnTransformer` for preprocessing (OneHotEncoder + StandardScaler)
- Supports both numerical and categorical feature handling

---

