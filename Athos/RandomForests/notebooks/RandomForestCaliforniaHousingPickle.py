#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import numpy as np
import pandas as pd


# In[2]:


# Import Dataset
dataset = pd.read_csv("housing.csv")
dataset.head()  # Print first 5 observations from dataset using head()


# In[3]:


# Check in which column contains nan values
dataset.isnull().any()


# In[5]:


# Separate features and labels
features = dataset.iloc[:, :-1].values
label = dataset.iloc[:, -1].values.reshape(-1, 1)
print("Sample feature vector: ", features[1])


# In[6]:


# Perform Imputation with strategy=mean
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer

imputerNaN = SimpleImputer(strategy="mean")
features[:, [4]] = imputerNaN.fit_transform(features[:, [4]])


# In[7]:


# Perform Label Encoding and Onehot Encoding on categorical values present in the features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

print("Features requiring encoding: ", set(features[:, 8]))
features[:, 8] = LabelEncoder().fit_transform(features[:, 8])
print("Sample feature vector: ", features[1])
print("No. of features: ", len(features[1]))

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [("Name", OneHotEncoder(), [8])], remainder="passthrough"
)  # The last arg ([0]) is the list of columns you want to transform in this step
features = ct.fit_transform(features).tolist()

# features = OneHotEncoder(categorical_features=[8]).fit_transform(features).toarray()
print("Sample feature vector after encoding: ", features[1])
print("No. of features after encoding: ", len(features[1]))


# In[15]:


X, y = (
    features,
    label,
)  # Purpose of this copying variables is that trees doesn't requires scaling while others "may be"
# Split into training set and testing set in every model building cause of "random_state" present in the "train_test_split"
from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=5)

# Random Forest Tree Regression
from sklearn.ensemble import RandomForestRegressor

depth = 9
no_of_estim = 10
model_random = RandomForestRegressor(
    n_estimators=no_of_estim, random_state=20, max_depth=depth
)
model_random.fit(X_train, y_train.ravel())

# Perform prediction and model score
y_pred = model_random.predict(X_test)
from sklearn.metrics import r2_score

print("Model Score for Training data: {}".format(model_random.score(X_train, y_train)))
print("Model Score for Testing data: {}".format(r2_score(y_test, y_pred)))
# print("Root Mean Squared Error is {}".format(np.sqrt(mean_squared_error(y_test,y_pred))))


# In[11]:


# Export pickle model
import pickle

pickle.dump(model_random, open("pickle_model.pickle", "wb"))


# In[16]:


test_input = np.array(X_test[0])
test_output = y_pred[0]

print("Test input: ", test_input)
print("Expected output: ", test_output)

with open("input.npy", "wb") as f:
    np.save(f, test_input)
print("Dumped input as np array in input.npy")


# In[ ]:
