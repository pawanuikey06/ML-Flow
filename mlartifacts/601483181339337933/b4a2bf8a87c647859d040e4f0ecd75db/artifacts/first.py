import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")
wine =load_wine()

X=wine.data
y=wine.target


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

max_depth =6
n_estimators=8

# Mention your experiment below
# mlflow.set_experiment('ML-Flow Learning')

# or instead set experiment we can directly pass the experiment id
with mlflow.start_run(experiment_id=601483181339337933 ):
    rf =RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(X_train,y_train)

    y_pred =rf.predict(X_test)

    accuracy =accuracy_score(y_test,y_pred)

    mlflow.log_metric('Accuracy',accuracy)
    mlflow.log_param('Max Depth',max_depth)
    mlflow.log_param('N Estimators',n_estimators)


    # creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

     # save plot
    plt.savefig("Confusion-matrix.png")

    # log artifacts using mlflow
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)


    print(accuracy)

