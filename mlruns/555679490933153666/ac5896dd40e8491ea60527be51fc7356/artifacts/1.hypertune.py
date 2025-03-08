from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow


# Load Breast Cancer Dataset
data=load_breast_cancer()
x=pd.DataFrame(data=data.data,columns=data.feature_names)
y=pd.Series(data.target,name='target')

# spilling into training and testing sets
X_train,X_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=42)

# creating the RandomForest classifier model

rf =RandomForestClassifier()


# Defining the parameter grid fo GridSearch
param_grid={
    'n_estimators':[10,50,100],
    'max_depth':[None,10,20,30]
}

# Applyin GridSeach
grid_search =GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs=1,verbose=2)


# # Running Without Mlflow
# grid_search.fit(X_train,y_train)


# # displaying the best param and best score

# best_param =grid_search.best_params_
# best_score=grid_search.best_score_


# print(best_param)
# print(best_score)

# With Mlflow

mlflow.set_experiment('Breast-Cancer-rf-hp')

with mlflow.start_run():
    # Run GridSearch
    grid_search.fit(X_train, y_train)

    # Get best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log parameters and score
    mlflow.log_params(best_params)
    mlflow.log_metric("best_score", best_score)

    # Log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")

    # Log source code (optional)
    mlflow.log_artifact(__file__)

    # Set tags
    mlflow.set_tag("author", "Pawan Kumar Uikey")

    # Print results
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)