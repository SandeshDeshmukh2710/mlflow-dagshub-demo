import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os

mlflow.set_tracking_uri("https://dagshub.com/SandeshDeshmukh2710/mlflow-dagshub-demo.mlflow")

import dagshub
dagshub.init(repo_owner='SandeshDeshmukh2710', repo_name='mlflow-dagshub-demo', mlflow=True)

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 10


mlflow.set_experiment('iris-dt')

with mlflow.start_run():

    dt = DecisionTreeClassifier(max_depth = max_depth)

    dt.fit(X_train, y_train) 

    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)

    mlflow.log_param("max_depth", max_depth)

    print("Accuracy :", accuracy)


     # Generate a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Save the heatmap as a file
    heatmap_path = "confusion_matrix_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()

    # Log the heatmap as an artifact
    mlflow.log_artifact(heatmap_path)

    mlflow.log_artifact(__file__)

    mlflow.set_tag('author','sandy')
    mlflow.set_tag('use','learning mlflow')

    mlflow.sklearn.log_model(dt, "Decison Tree")
