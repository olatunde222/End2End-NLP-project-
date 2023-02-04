#!/usr/bin/env python
# coding: utf-8

# ### End 2 End NLP Project 
# + Emotion Detection In Text 
#     - Text Classifier 
#   

# **This project is intended to detect the posibile emotion behind a text. This project will be in the stages below:**  
# + Loading all the dependencies
# + Data Preprocessing 
# + Model Building 
# + Model Evaluation 
# + Model Deployment to Web App Using the Streamlit library 

# ### Importing all the neccessary dependencies 

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import neattext as nfx
from matplotlib import pyplot as plt 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ### Data Preprocessing 
# 

# In[2]:


path = 'Emotion dataset.csv'
data = pd.read_csv(path)


# In[3]:


data.head()


# In[4]:


# Dropping the [Unnamed: 3] column
data.drop(["Unnamed: 3","Unnamed: 0"],axis=1,inplace=True)


# In[5]:


#Dropping the missing values 
data.dropna(inplace=True)


# In[6]:


#Counting the Emotions Classifications in the data 

data["Emotion"].value_counts().plot(kind="bar")


# In[7]:


# Cleaning the text data
data["Cleaned_Text"] = data["Text"].apply(nfx.remove_userhandles)
data["Cleaned_Text"] = data["Text"].apply(nfx.remove_stopwords)


# In[8]:


data.head(3)


# ### Model Building 

# In[9]:


Xfeatures = data["Cleaned_Text"]
Ylabels = data["Emotion"]

# Splitting the data for training and testing 

Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xfeatures,Ylabels,test_size=0.3, random_state=42)


# In[10]:


print(Xfeatures.shape,Xtrain.shape,Xtest.shape)
print(Ylabels.shape,Ytrain.shape,Ytest.shape)


# In[13]:


# Building a pipeline for the Model 
from sklearn.pipeline import Pipeline

# Logistic regression 
pipe_lr = Pipeline(steps=[("cv",CountVectorizer()), ("lr",LogisticRegression())])


# In[14]:


# Training and Fitting the data 

pipe_lr.fit(Xtrain,Ytrain)


# In[15]:


# Accuracy 

pipe_lr.score(Xtest,Ytest)


# In[16]:


# Making a single prediction 

example = "This book was so interesting it made my day"

pipe_lr.predict([example])


# In[17]:


# Prediction probability 

pipe_lr.predict_proba([example])


# In[18]:


pipe_lr.predict_proba([example]).max()


# In[19]:


pipe_lr.classes_


# ### saving the model 

# In[20]:


import pickle 

file_name = "End2End_NLP_Project_Emotion_Classifier_pipe_lr_02_feb_23"

pickle.dump(pipe_lr,open(file_name,'wb'))


# ### Deploying and productionizing the model into App Using streamlit 

# In[ ]:





# In[ ]:




