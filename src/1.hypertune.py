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


# Running Without Mlflow
grid_search.fit(X_train,y_train)


# displaying the best param and best score

best_param =grid_search.best_params_
best_score=grid_search.best_score_


print(best_param)
print(best_score)
