# Student Performance Prediction

Predicts a student's final exam score using Linear Regression, based on
study hours, attendance, previous exam score, and sleep hours.

## Project structure
```
student-performance-prediction/
├── train_model.py     # generates data, trains the model, saves it
├── requirements.txt
├── data/
│   └── student_data.csv           (created when you run train_model.py)
├── model/
│   ├── linear_regression_model.pkl (created when you run train_model.py)
│   └── actual_vs_predicted.png     (created when you run train_model.py)
└── app.py              # Streamlit web app (coming in Step 3)
```

## How to run
```
pip install -r requirements.txt
python train_model.py
```
This creates the dataset, trains the model, and saves the trained model
and a diagnostic plot into `model/`.

(Web app instructions will be added in Step 3.)
