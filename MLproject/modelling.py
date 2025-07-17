from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import sys
import argparse
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # ambil argumen CLI 
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "personality_preprocessing_datasert.csv")
    test_path = file_path.replace("personality_preprocessing_datasert.csv", "data_test_personality.csv")

    print(f"Reading data train from : {file_path}")
    print(f"Reading data test from : {test_path}")

    # load data
    train_df = pd.read_csv(file_path)
    test_df = pd.read_csv(test_path) 
    target = 'Personality'

    X_train = train_df.drop(columns=target)
    y_train = train_df[target]
    X_test = test_df.drop(columns=target)
    y_test = test_df[target]

    # set tracking URI 
    if "GITHUB_ACTIONS" in os.environ:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
    else:
        mlflow.set_tracking_uri("file://"+ os.path.abspath("mlruns"))
    
    with mlflow.start_run():
        param = {'n_neighbors': np.arange(3,4,5)}
        kf = KFold(n_splits=3, shuffle=True)

        model = KNeighborsClassifier()

        grid_search = GridSearchCV(model, param_grid=param, cv=kf, verbose=1) 

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # 🔒 Manual log
        mlflow.log_param("best_n_neighbors", grid_search.best_params_['n_neighbors'])
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        mlflow.log_metric("test_accuracy", acc)

        # Optional: log classification report per class
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for m_name, m_val in metrics.items():
                    mlflow.log_metric(f"{label}_{m_name}", m_val)
        
        # log model
        print("run_id=", mlflow.active_run().info.run_id)
        mlflow.sklearn.log_model(best_model, artifact_path="model", input_example=X_train.iloc[:1])


        print("Best params :", grid_search.best_params_)
        print("Best cv Score :", grid_search.best_score_)
        print("Accuracy Test :", acc)
        print("Confusion Matrix:\n", cm)

