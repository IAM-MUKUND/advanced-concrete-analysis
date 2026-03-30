#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv("combined_concrete_data.csv")

df.head()


# In[ ]:


df.describe().T


# I am going to train with the models, which innately support NANs in the dataset(lightgbm, xgboost), and then impute 0 valuse to the NANs and train other models.

# In[ ]:


target = "compressive strength"

X = df.drop(columns=[target])
y = df[target]


# In[ ]:


from sklearn.model_selection import train_test_split

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=df['concrete type'] 
)


# In[ ]:


print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# In[ ]:


import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')


# In[ ]:


# implementing standard scaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# In[ ]:


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor


# In[ ]:


from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

imputer = SimpleImputer(strategy='mean')

X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train))
X_test_imputed = pd.DataFrame(imputer.fit_transform(X_test))


# In[ ]:


# XGB REGRESSOR

xgb_model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    missing=np.nan,           
    enable_categorical=True   
)

param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

best_xgb_model = grid_search.best_estimator_
print("Best params:", grid_search.best_params_)


# In[ ]:


# LGBM REGRESSOR

import optuna
from sklearn.model_selection import cross_val_score, KFold
import logging

def objective(trial):
    params = {
        # Search space
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1 # Let LightGBM use all cores
    }
    
    model = LGBMRegressor(**params)
    
    # Example using 5-fold CV for speed during tuning
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error', n_jobs=1)
    
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=150, show_progress_bar=False) 

print("Best params:", study.best_params)


# I implemented the optuna hyperparameter tuning for lightgbm, and it is giving me a good boost in the performance. I will implement it for xgboost as well, and then move on to the imputation part. I implemented the optuna hyperparameter tuning, because the gridsearchcv was taking a lot of time to run, like 5 candidates initialization in gridsearchcv was taking around 5.37 ish minutes, and I had around 125+ candidates to try. So optune is a good choice for hyperparameter tuning in this case, because it is much faster than gridsearchcv, and it also gives better results. I will not implement it to others (for now), if situation arises, I will turn to optuna.

# In[ ]:


# CATBOOST REGRESSOR

from catboost import CatBoostRegressor

catboost_model = CatBoostRegressor(random_state=42, verbose=0)

param_grid_catboost = {'iterations': [500, 800], 'depth': [6,8,10],
              'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0]}

grid = GridSearchCV(catboost_model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
best_catboost_model = grid.best_estimator_


# In[ ]:


# GRADIENT BOOSTING REGRESSOR

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(random_state=42)

param_grid_gbr = {'n_estimators': [300, 500], 'max_depth': [5,7],
              'learning_rate': [0.05, 0.1]}

grid = GridSearchCV(model, param_grid_gbr, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid.fit(X_train_imputed, y_train)
best_model_gradient_boost = grid.best_estimator_


# In[ ]:


# RANDOMFOREST REGRESSOR

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42, n_jobs=-1)

param_grid_rf = {'n_estimators': [200, 300], 'max_depth': [10, 15, 20],
              'min_samples_split': [2, 5]}

grid = GridSearchCV(model, param_grid_rf, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid.fit(X_train_imputed, y_train)
best_model_rf = grid.best_estimator_

y_pred_rf = best_model_rf.predict(X_test_imputed)



# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2 Score": r2_score(y_true, y_pred)
    }


# In[ ]:


y_pred_xgb = best_xgb_model.predict(X_test)

xgb_eval = evaluate_model(y_test, y_pred_xgb)

print(xgb_eval)


# In[ ]:


y_pred_catboost = best_catboost_model.predict(X_test)

catboost_eval = evaluate_model(y_test, y_pred_catboost)

print(catboost_eval)


# In[ ]:


best_lgbm_params = study.best_params

lgbm_model = LGBMRegressor(
    boosting_type="gbdt",
    random_state=42,
    verbose=-1,
    **best_lgbm_params
)

lgbm_model.fit(X_train, y_train)

y_pred_lgbm = lgbm_model.predict(X_test)

lgbm_eval = evaluate_model(y_test, y_pred_lgbm)

print(lgbm_eval)


# In[ ]:


y_pred_gradient_boost = best_model_gradient_boost.predict(X_test_imputed)

gradient_boost_eval = evaluate_model(y_test ,y_pred_gradient_boost)

print(gradient_boost_eval)


# In[ ]:




