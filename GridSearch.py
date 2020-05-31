# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import joblib
from joblib import parallel


# %%
# Load Data
df = pd.read_csv("ml_house_data_set.csv")


# %%
# Removing insignificant features
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']


# %%
# Replacing categorical data with one-hot encoded data
features_df = pd.get_dummies(df, columns = ['garage_type', 'city'])


# %%
del features_df['sale_price']


# %%
# Converting features into matrix / Array form 
X = features_df.values
y = df['sale_price'].values


# %%
# Splitting data into training set (70%) and test data (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# %%
# Create a model
model = ensemble.GradientBoostingRegressor()


# %%
model


# %%
# Parameters of the model 
param_grid = {
    'n_estimators': [500,1000,3000],
    'max_depth': [4,6],
    'min_samples_leaf': [3,5,9,17],
    'learning rate': [0.1, 0.05, 0.02, 0.01],
    'max_features': [1.0, 0.3, 0.1],
    'loss': ['ls', 'lad', 'huber'] 
}


# %%
param_grid


# %%
# Define grid search and run 4 jobs
gs_cv = GridSearchCV(model, param_grid, n_jobs = 4)


# %%
gs_cv


# %%
# Run Grid Search in training data
gs_cv.fit(X_train, y_train)


# %%
# Print Parameters that givce best result
print(gs_cv.best_params_)


# %%
# Finding the error rate on the training set using the best parameters
mse = mean_absolute_error(y_train, gs_cv.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)


# %%
# Finding the error rate on the test set using the best parameters
mse = mean_absolute_error(y_test, gs_cv.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse )
