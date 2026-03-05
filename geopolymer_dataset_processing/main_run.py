#!/usr/bin/env python
# coding: utf-8

# Data Preprocessing

# In[1]:


import pandas as pd
import numpy as np 


# In[2]:


df = pd.read_excel("GEOPOLYMER_CONCRETE_24_2_24.xlsx", header=[0,1,2,3])
df.head()


# In[3]:


df.tail()


# In[4]:


df.shape


# In[5]:


df_summary = df.tail(3).copy()
df_summary


# In[6]:


df_summary = df.tail(3).copy()

main_df = df[pd.to_numeric(df.iloc[:, 2], errors='coerce').notnull()]

main_df = main_df.dropna(how='all', axis=1)

print(f"Cleaned dataset shape: {main_df.shape}")
print(f"Summary stats shape: {df_summary.shape}")


# In[7]:


main_df.head()


# In[8]:


main_df.tail()


# In[9]:


main_df = main_df.dropna(subset=[main_df.columns[3]])

main_df.tail()


# In[10]:


main_df.columns = ["s.no", "authors", "number", "binders", "extra water", "alkaline solution", "molarity of mix", "fine aggregate", "coarse aggregate", "age", "curing temperature", "compressive strength"]

main_df.columns


# In[11]:


main_df.head()


# In[12]:


main_df = main_df.iloc[:, 3:]
print(f"Final shape for ML: {main_df.shape}")
main_df.head()


# In[13]:


main_df.isnull().sum()


# In[14]:


main_df.duplicated().sum()


# In[15]:


main_df.drop_duplicates(inplace=True)


# In[16]:


main_df.shape


# EDA- Exploratory Data Analysis

# In[17]:


main_df.info()


# In[18]:


main_df.describe().T


# In[19]:


main_df.nunique()


# In[20]:


#multivariate analysis

import seaborn as sns

sns.heatmap(main_df.corr(), annot=True, fmt='.2f', cmap='Pastel2', linewidths=2)


# Dataset preparation

# In[21]:


X = main_df.drop("compressive strength", axis=1)
y = main_df["compressive strength"]

X.shape,y.shape


# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[23]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model Training

# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor


# In[25]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2 Score": r2_score(y_true, y_pred)
    }


# In[26]:


#Linear Regression

lr = LinearRegression()
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

lr_evaluation = evaluate_model(y_test, pred_lr)


# In[27]:


#Decision Tree Regressor

dt = DecisionTreeRegressor(
    max_depth=6,
    random_state=42
)

dt.fit(X_train, y_train)

pred_dt = dt.predict(X_test)

dt_evaluation = evaluate_model(y_test, pred_dt)


# In[28]:


#Random Forest

rf = RandomForestRegressor(n_estimators=200)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

rf_evaluation = evaluate_model(y_test, pred_rf)


# In[29]:


#Support Vector Regression

svm = SVR()

svm.fit(X_train, y_train)

pred_svm = svm.predict(X_test)

svm_evaluation = evaluate_model(y_test, pred_svm)


# In[30]:


#K-Nearest Neighbors

knn = KNeighborsRegressor(n_neighbors=5)

knn.fit(X_train, y_train)

pred_knn = knn.predict(X_test)

knn_evaluation = evaluate_model(y_test, pred_knn)


# In[31]:


#XGBoost Regressor

xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

xgb.fit(X_train, y_train)

pred_xgb = xgb.predict(X_test)

xgb_evaluation = evaluate_model(y_test, pred_xgb)


# In[32]:


# Artificial Neural Network

ann = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
)

ann.fit(X_train, y_train)

y_pred = ann.predict(X_test)

ann_evaluation = evaluate_model(y_test, y_pred)


# Evaluation of the models

# MAE: MAE calculates the absolute difference between the actual value and the predicted value, and then takes the average of all these differences. It tells you, on average, how far off your predictions are from the true values, regardless of whether you over-predicted or under-predicted. Lower is better
# 
# MSE:  MSE calculates the difference between the actual value and the predicted value, squares that difference, and then takes the average across all data points. Because you are squaring the errors, MSE heavily penalizes large errors. If your model is mostly accurate but makes a few massive mistakes (outliers), the MSE will skyrocket. The unit is the square of your target variable's unit (e.g., MPa²), which makes it hard to interpret directly. Lower is better
# 
# RMSE is simply the square root of the MSE. Taking the square root converts the metric back into the original units of your target variable (e.g., MPa), making it easier to interpret exactly like MAE. However, because it is derived from MSE, it inherits the property of heavily penalizing large errors/outliers. Lower is better.
# 
# R squared score: Unlike the other three which measure error, R² measures the goodness of fit. It represents the proportion of the variance in the target variable (compressive strength) that can be explained by your model's features. Intuition: It is a relative metric compared to a naive "baseline" model. If you built a stupid model that just predicted the exact average of all compressive strengths for every single row, that baseline model would get an R² of 0. Higher is better

# In[33]:


results = pd.DataFrame([
    {
        'Model': 'Linear Regression',
        'R2 Score': lr_evaluation['R2 Score'],
        'MAE': lr_evaluation['MAE'],
        'MSE': lr_evaluation['MSE'],
        'RMSE': lr_evaluation['RMSE']
    },
    {
        'Model': 'Decision Tree',
        'R2 Score': dt_evaluation['R2 Score'],
        'MAE': dt_evaluation['MAE'],
        'MSE': dt_evaluation['MSE'],
        'RMSE': dt_evaluation['RMSE']
    },
    {
        'Model': 'Random Forest',
        'R2 Score': rf_evaluation['R2 Score'],
        'MAE': rf_evaluation['MAE'],
        'MSE': rf_evaluation['MSE'],
        'RMSE': rf_evaluation['RMSE']
    },
    {
        'Model': 'Support Vector Regression',
        'R2 Score': svm_evaluation['R2 Score'],
        'MAE': svm_evaluation['MAE'],
        'MSE': svm_evaluation['MSE'],
        'RMSE': svm_evaluation['RMSE']
    },
    {
        'Model': 'K-Nearest Neighbors',
        'R2 Score': knn_evaluation['R2 Score'],
        'MAE': knn_evaluation['MAE'],
        'MSE': knn_evaluation['MSE'],
        'RMSE': knn_evaluation['RMSE']
    },
    {
        'Model': 'XGBoost',
        'R2 Score': xgb_evaluation['R2 Score'],
        'MAE': xgb_evaluation['MAE'],
        'MSE': xgb_evaluation['MSE'],
        'RMSE': xgb_evaluation['RMSE']
    },
    {
        'Model': 'ANN',
        'R2 Score': ann_evaluation['R2 Score'],
        'MAE': ann_evaluation['MAE'],
        'MSE': ann_evaluation['MSE'],
        'RMSE': ann_evaluation['RMSE']
    }
])


# Visualization

# In[34]:


import matplotlib.pyplot as plt


# In[35]:


#RMSE Comparison 

plt.figure(figsize=(10, 6))
plt.bar(results['Model'], results['RMSE'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('RMSE Comparison')
plt.xticks(rotation=45)
plt.show()


# In[36]:


#MSE Comparison

plt.figure(figsize=(10, 6))
plt.bar(results['Model'], results['MSE'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('MSE')
plt.title('MSE Comparison')
plt.xticks(rotation=45)
plt.show()


# In[37]:


#MAE Comparison

plt.figure(figsize=(10,6))
plt.bar(results['Model'], results['MAE'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('MAE')
plt.title('MAE Comparison')
plt.xticks(rotation=45)
plt.show()


# In[38]:


#R2 SCORE Comparison

plt.figure(figsize=(10, 6))
plt.bar(results['Model'], results['R2 Score'], color='lightgreen')
plt.xlabel('Model')
plt.ylabel('R2 Score')
plt.title('R2 Score Comparison')
plt.xticks(rotation=45)
plt.show()


# Linear Regression and Support Vector Regression are showing weak performance and they struggle to model this dataset
# 
# Decision tree and K - Nearest Neighbors Regression are showing good performance and they are able to model this dataset well.
# 
# XGBoost and Random Forest Regression is showing the best performance and it is able to model this dataset VERY WELL.
# 
# Now analysing the feature importance through Random Forest Regression and XGBoost.

# In[39]:


#feature importance of the random forest regression model.

rf_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
})

rf_importance = rf_importance.sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=rf_importance, palette='viridis', hue='feature', legend=False)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Predicting Compressive Strength')
plt.show()

print("\nFeature Importances:")
print(rf_importance)


# In[40]:


#Feature importance of the XGBoost regression model.

xgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb.feature_importances_
})

xgb_importance = xgb_importance.sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=xgb_importance, palette='viridis', hue='feature', legend=False)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Predicting Compressive Strength')
plt.show()

print("\nFeature Importances:")
print(xgb_importance)


# In[ ]:




print(results.to_string())
