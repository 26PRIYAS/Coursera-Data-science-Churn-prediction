#!/usr/bin/env python
# coding: utf-8

# ![COUR_IPO.png](attachment:COUR_IPO.png)

# # Welcome to the Data Science Coding Challange!
# 
# Test your skills in a real-world coding challenge. Coding Challenges provide CS & DS Coding Competitions with Prizes and achievement badges!
# 
# CS & DS learners want to be challenged as a way to evaluate if they’re job ready. So, why not create fun challenges and give winners something truly valuable such as complimentary access to select Data Science courses, or the ability to receive an achievement badge on their Coursera Skills Profile - highlighting their performance to recruiters.

# ## Introduction
# 
# In this challenge, you'll get the opportunity to tackle one of the most industry-relevant maching learning problems with a unique dataset that will put your modeling skills to the test. Subscription services are leveraged by companies across many industries, from fitness to video streaming to retail. One of the primary objectives of companies with subscription services is to decrease churn and ensure that users are retained as subscribers. In order to do this efficiently and systematically, many companies employ machine learning to predict which users are at the highest risk of churn, so that proper interventions can be effectively deployed to the right audience.
# 
# In this challenge, we will be tackling the churn prediction problem on a very unique and interesting group of subscribers on a video streaming service! 
# 
# Imagine that you are a new data scientist at this video streaming company and you are tasked with building a model that can predict which existing subscribers will continue their subscriptions for another month. We have provided a dataset that is a sample of subscriptions that were initiated in 2021, all snapshotted at a particular date before the subscription was cancelled. Subscription cancellation can happen for a multitude of reasons, including:
# * the customer completes all content they were interested in, and no longer need the subscription
# * the customer finds themselves to be too busy and cancels their subscription until a later time
# * the customer determines that the streaming service is not the best fit for them, so they cancel and look for something better suited
# 
# Regardless the reason, this video streaming company has a vested interest in understanding the likelihood of each individual customer to churn in their subscription so that resources can be allocated appropriately to support customers. In this challenge, you will use your machine learning toolkit to do just that!

# ## Understanding the Datasets

# ### Train vs. Test
# In this competition, you’ll gain access to two datasets that are samples of past subscriptions of a video streaming platform that contain information about the customer, the customers streaming preferences, and their activity in the subscription thus far. One dataset is titled `train.csv` and the other is titled `test.csv`.
# 
# `train.csv` contains 70% of the overall sample (243,787 subscriptions to be exact) and importantly, will reveal whether or not the subscription was continued into the next month (the “ground truth”).
# 
# The `test.csv` dataset contains the exact same information about the remaining segment of the overall sample (104,480 subscriptions to be exact), but does not disclose the “ground truth” for each subscription. It’s your job to predict this outcome!
# 
# Using the patterns you find in the `train.csv` data, predict whether the subscriptions in `test.csv` will be continued for another month, or not.

# ### Dataset descriptions
# Both `train.csv` and `test.csv` contain one row for each unique subscription. For each subscription, a single observation (`CustomerID`) is included during which the subscription was active. 
# 
# In addition to this identifier column, the `train.csv` dataset also contains the target label for the task, a binary column `Churn`.
# 
# Besides that column, both datasets have an identical set of features that can be used to train your model to make predictions. Below you can see descriptions of each feature. Familiarize yourself with them so that you can harness them most effectively for this machine learning task!

# In[1]:


import pandas as pd
data_descriptions = pd.read_csv('data_descriptions.csv')
pd.set_option('display.max_colwidth', None)
data_descriptions


# ## How to Submit your Predictions to Coursera
# Submission Format:
# 
# In this notebook you should follow the steps below to explore the data, train a model using the data in `train.csv`, and then score your model using the data in `test.csv`. Your final submission should be a dataframe (call it `prediction_df` with two columns and exactly 104,480 rows (plus a header row). The first column should be `CustomerID` so that we know which prediction belongs to which observation. The second column should be called `predicted_probability` and should be a numeric column representing the __likellihood that the subscription will churn__.
# 
# Your submission will show an error if you have extra columns (beyond `CustomerID` and `predicted_probability`) or extra rows. The order of the rows does not matter.
# 
# The naming convention of the dataframe and columns are critical for our autograding, so please make sure to use the exact naming conventions of `prediction_df` with column names `CustomerID` and `predicted_probability`!
# 
# To determine your final score, we will compare your `predicted_probability` predictions to the source of truth labels for the observations in `test.csv` and calculate the [ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html). We choose this metric because we not only want to be able to predict which subscriptions will be retained, but also want a well-calibrated likelihood score that can be used to target interventions and support most accurately.

# ## Import Python Modules
# 
# First, import the primary modules that will be used in this project. Remember as this is an open-ended project please feel free to make use of any of your favorite libraries that you feel may be useful for this challenge. For example some of the following popular packages may be useful:
# 
# - pandas
# - numpy
# - Scipy
# - Scikit-learn
# - keras
# - maplotlib
# - seaborn
# - etc, etc

# In[ ]:


# Import required packages

# Data packages
import pandas as pd
import numpy as np

# Machine Learning / Classification packages
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# Visualization Packages
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Import any other packages you may want to use


# ## Load the Data
# 
# Let's start by loading the dataset `train.csv` into a dataframe `train_df`, and `test.csv` into a dataframe `test_df` and display the shape of the dataframes.

# In[ ]:


import pandas as pd
train_df = pd.read_csv("train.csv")
print('train_df Shape:', train_df.shape)
train_df.head() 


# In[ ]:


import pandas as pd
test_df = pd.read_csv("test.csv")
print('test_df Shape:', test_df.shape)
test_df.head()


# ## Explore, Clean, Validate, and Visualize the Data (optional)
# 
# Feel free to explore, clean, validate, and visualize the data however you see fit for this competition to help determine or optimize your predictive model. Please note - the final autograding will only be on the accuracy of the `prediction_df` predictions.

# In[ ]:


train_df.drop("CustomerID", axis=1, inplace=True)
train_df.head()
train_df["Churn"].value_counts()
train_df.isna().sum()


# In[ ]:


train_df["Churn"].value_counts().plot(kind='bar', title='Churn Counts');


# In[ ]:


noChurn = train_df[train_df["Churn"] == 0]
yesChurn = train_df[train_df["Churn"] == 1]


# In[ ]:


# define yes or no column
yesNoColumns = ["PaperlessBilling", "MultiDeviceAccess", "ParentalControl", "SubtitlesEnabled"]
# define several columns for one hot encoding
columnsForOneHotEncoding = ["SubscriptionType", "PaymentMethod", "ContentType", "DeviceRegistered", "GenrePreference"]
# define the columbns to be scaled
columnsToScale = ["AccountAge","MonthlyCharges", "TotalCharges", "ViewingHoursPerWeek", "AverageViewingDuration", 
                  "ContentDownloadsPerMonth", "UserRating", "SupportTicketsPerMonth" ]

# for col in yesNoColumns:
#     train_df[col].replace({'Yes': 1, 'No': 0}, inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


len(yesChurn) , len(noChurn) # seem to be imbalanced , yes churn is less than no churn


# In[ ]:


sampledNoChurn = noChurn.sample(n=len(yesChurn), random_state=42)
sampledNoChurn.shape
(44182, 20)


# In[ ]:


balanceTrainDF = pd.concat([sampledNoChurn, yesChurn], axis=0)
balanceTrainDF.shape


# In[ ]:


balanceTrainDF = preprocessData(balanceTrainDF) 


# In[ ]:


test_df= preprocessData(test_df)
test_df.sample(5)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
def preprocessData(df, scale=True):

    # Convert Yes/No columns to 1/0
    for col in yesNoColumns:
        df[col].replace({'Yes': 1, 'No': 0}, inplace=True)
        
    # Convert gender to 1/0
    df.Gender.replace({"Male": 1, "Female": 0}, inplace=True)

    # One hot encoding
    df = pd.get_dummies(df, columns=columnsForOneHotEncoding, dtype=int)

    if scale:
        # Scale columns
        scaler = MinMaxScaler()#StandardScaler()
        df[columnsToScale] = scaler.fit_transform(df[columnsToScale])

    return df


# ## Make predictions (required)
# 
# Remember you should create a dataframe named `prediction_df` with exactly 104,480 entries plus a header row attempting to predict the likelihood of churn for subscriptions in `test_df`. Your submission will throw an error if you have extra columns (beyond `CustomerID` and `predicted_probaility`) or extra rows.
# 
# The file should have exactly 2 columns:
# `CustomerID` (sorted in any order)
# `predicted_probability` (contains your numeric predicted probabilities between 0 and 1, e.g. from `estimator.predict_proba(X, y)[:, 1]`)
# 
# The naming convention of the dataframe and columns are critical for our autograding, so please make sure to use the exact naming conventions of `prediction_df` with column names `CustomerID` and `predicted_probability`!

# In[ ]:


## Install  xgboost and lightgbm
get_ipython().system('pip install xgboost')
get_ipython().system('pip install lightgbm')


# In[22]:


#import ML different model
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


# In[23]:


nEstimate=  200, #850#1000#500
rounds=10
maxDepth = 7#10

rf = RandomForestClassifier(n_estimators=200, 
                            class_weight='balanced', 
                            random_state=42, n_jobs=-1)


# In[24]:


nEstimate=300 #1000
rounds=10
maxDepth = 6#10

xgb = XGBClassifier(n_estimators=nEstimate,
                     max_depth=maxDepth, 
                     objective='binary:logistic',
                    #  tree_method='gpu_hist', # GPU accelerated training.
                     n_jobs=-1, # for CPU parallelism
                     random_state=42)
xgb.get_params()


# ### Example prediction submission:
# 
# The code below is a very naive prediction method that simply predicts churn using a Dummy Classifier. This is used as just an example showing the submission format required. Please change/alter/delete this code below and create your own improved prediction methods for generating `prediction_df`.

# In[25]:


lgbmModel = LGBMClassifier(n_estimators=nEstimate, 
                           max_depth=maxDepth, 
                           n_jobs=-1,
                           class_weight='balanced',
                           random_state=42)


# In[26]:


# X = train_df.drop("Churn", axis=1)
# y = train_df["Churn"]
X = balanceTrainDF.drop("Churn", axis=1)
y = balanceTrainDF["Churn"]


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# In[28]:


X_train.shape, X_test.shape


# In[42]:


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Define the categorical columns
categorical_columns = ['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'ContentType', 
                       'MultiDeviceAccess', 'DeviceRegistered', 'GenrePreference', 
                       'Gender', 'ParentalControl', 'SubtitlesEnabled']

# Apply One-Hot Encoding
encoder = OneHotEncoder(drop='first')  # drop='first' to avoid multicollinearity
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_columns]).toarray())

# Combine with the rest of the numerical columns
X_train_numeric = X_train.drop(categorical_columns, axis=1)
X_train_final = pd.concat([X_train_numeric, X_train_encoded], axis=1)


# In[43]:


from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Loop through the categorical columns and encode them
for column in categorical_columns:
    X_train[column] = label_encoder.fit_transform(X_train[column])


# In[41]:


X_train.dtypes  # This will show the data type of each column


# In[49]:


rf.fit(X_train, y_train)


# In[50]:


X_test.dtypes  # This will show the data type of each column


# In[57]:


from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Define the categorical columns
categorical_columns = ['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 
                       'ContentType', 'MultiDeviceAccess', 'DeviceRegistered', 
                       'GenrePreference', 'Gender', 'ParentalControl', 
                       'SubtitlesEnabled']

# Apply One-Hot Encoding
encoder = OneHotEncoder(drop='first')  # drop='first' to avoid multicollinearity
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_test[categorical_columns]).toarray())

# Combine with the rest of the numerical columns
X_test_numeric = X_test.drop(categorical_columns, axis=1)
X_test_final = pd.concat([X_test_numeric, X_test_encoded], axis=1)


# In[58]:


from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Loop through the categorical columns and encode them
for column in categorical_columns:
    X_test[column] = label_encoder.fit_transform(X_test[column])


# In[59]:


evalRF = rf.predict(X_test)
evalRFProb = rf.predict_proba(X_test)[:,1] # get the probability of the positive class


# In[60]:


from sklearn.metrics import classification_report

reportClassificationRF = classification_report(y_test, evalRF)
print(reportClassificationRF)


# In[62]:


from sklearn.metrics import roc_curve, roc_auc_score

rfc_auc = roc_auc_score(y_test, evalRFProb)
rfc_fpr, rfc_tpr, rfc_th = roc_curve(y_test, evalRFProb)
rfc_auc


# In[63]:


plt.figure(figsize=(6, 4))
plt.plot(rfc_fpr, rfc_tpr, label='ROC curve (area = %0.2f)' % rfc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest Classification')
plt.legend()
plt.show()


# In[69]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# List of categorical columns to encode
categorical_columns = ['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 
                       'ContentType', 'MultiDeviceAccess', 'DeviceRegistered', 
                       'GenrePreference', 'Gender', 'ParentalControl', 
                       'SubtitlesEnabled']

# Apply LabelEncoder to each categorical column in test_df
for col in categorical_columns:
    test_df[col] = label_encoder.fit_transform(test_df[col])

# Now you can safely make predictions
predicted_probability = rf.predict_proba(test_df.drop(["CustomerID"], axis=1))[:,1]


# In[70]:


### PLEASE CHANGE THIS CODE TO IMPLEMENT YOUR OWN PREDICTIONS

# Combine predictions with label column into a dataframe
prediction_df = pd.DataFrame({'CustomerID': test_df[['CustomerID']].values[:, 0],
                             'predicted_probability': predicted_probability})


# In[71]:


### PLEASE CHANGE THIS CODE TO IMPLEMENT YOUR OWN PREDICTIONS

# View our 'prediction_df' dataframe as required for submission.
# Ensure it should contain 104,480 rows and 2 columns 'CustomerID' and 'predicted_probaility'
print(prediction_df.shape)
prediction_df.head(10)


# **PLEASE CHANGE CODE ABOVE TO IMPLEMENT YOUR OWN PREDICTIONS**

# ## Final Tests - **IMPORTANT** - the cells below must be run prior to submission
# 
# Below are some tests to ensure your submission is in the correct format for autograding. The autograding process accepts a csv `prediction_submission.csv` which we will generate from our `prediction_df` below. Please run the tests below an ensure no assertion errors are thrown.

# In[72]:


# FINAL TEST CELLS - please make sure all of your code is above these test cells

# Writing to csv for autograding purposes
prediction_df.to_csv("prediction_submission.csv", index=False)
submission = pd.read_csv("prediction_submission.csv")

assert isinstance(submission, pd.DataFrame), 'You should have a dataframe named prediction_df.'


# In[73]:


# FINAL TEST CELLS - please make sure all of your code is above these test cells

assert submission.columns[0] == 'CustomerID', 'The first column name should be CustomerID.'
assert submission.columns[1] == 'predicted_probability', 'The second column name should be predicted_probability.'


# In[74]:


# FINAL TEST CELLS - please make sure all of your code is above these test cells

assert submission.shape[0] == 104480, 'The dataframe prediction_df should have 104480 rows.'


# In[75]:


# FINAL TEST CELLS - please make sure all of your code is above these test cells

assert submission.shape[1] == 2, 'The dataframe prediction_df should have 2 columns.'


# In[76]:


# FINAL TEST CELLS - please make sure all of your code is above these test cells

## This cell calculates the auc score and is hidden. Submit Assignment to see AUC score.


# ## SUBMIT YOUR WORK!
# 
# Once we are happy with our `prediction_df` and `prediction_submission.csv` we can now submit for autograding! Submit by using the blue **Submit Assignment** at the top of your notebook. Don't worry if your initial submission isn't perfect as you have multiple submission attempts and will obtain some feedback after each submission!

# In[ ]:




