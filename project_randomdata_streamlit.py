#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df=pd.read_csv("D:/Harini(christ unniversity)/3rd sem subjects/Project/2048034_55_Dataset.csv")


# In[4]:


df.shape


# In[5]:


data = df.sample(n=1000000)


# In[7]:


data.to_csv('D:/Harini(christ unniversity)/3rd sem subjects/Project/randomly_select_data.csv', index=True, encoding='utf-8')


# In[8]:


data.shape


# In[12]:


data.info()


# In[13]:


data.describe()


# In[14]:


data.isnull().values.any()


# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import warnings
warnings.filterwarnings("ignore")
import streamlit as st


# In[10]:


st.title('Financial Fraud Detection!')


# In[11]:


df=st.cache(pd.read_csv)('D:/Harini(christ unniversity)/3rd sem subjects/Project/randomly_select_data.csv')
#df = df.sample(frac=0.1, random_state = 48)


# In[15]:


# Print shape and description of the data
if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(data.head(10))
    st.write('Shape of the dataframe: ',data.shape)
    st.write('Data decription: \n',data.describe())
# Print valid and fraud transactions
fraud=data[data.isFraud==1]
valid=data[data.isFraud==0]
outlier_percentage=(data.isFraud.value_counts()[1]/data.isFraud.value_counts()[0])*100
if st.sidebar.checkbox('Show fraud and valid transaction details'):
    st.write('Fraudulent transactions are: %.3f%%'%outlier_percentage)
    st.write('Fraud Cases: ',len(fraud))
    st.write('Valid Cases: ',len(valid))


# In[16]:


from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[18]:


data.type = le.fit_transform(data.type)


# In[19]:


data.info()


# In[21]:


#Obtaining X (features) and y (labels)
X=data.drop(['isFraud','nameOrig','nameDest','isFlaggedFraud'], axis=1)
y=data.isFraud
#Split the data into training and testing sets
from sklearn.model_selection import train_test_split
size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 42)
#Print shape of train and test sets
if st.sidebar.checkbox('Show the shape of training and test set features and labels'):
    st.write('X_train: ',X_train.shape)
    st.write('y_train: ',y_train.shape)
    st.write('X_test: ',X_test.shape)
    st.write('y_test: ',y_test.shape)


# In[22]:


#Import classification models and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
logreg=LogisticRegression()
svm=SVC()
knn=KNeighborsClassifier()
etree=ExtraTreesClassifier(random_state=42)
rforest=RandomForestClassifier(random_state=42)


# In[23]:


X_train.info()


# In[28]:


#Feature selection through feature importance
#@st.cache
def feature_sort(model,X_train,y_train):
    #feature selection
    mod=model
    # fit the model
    mod.fit(X_train, y_train)
    # get importance
    imp = mod.feature_importances_
    return imp
#Classifiers for feature importance

clf=['Extra Trees','Random Forest']
mod_feature = st.sidebar.selectbox('Which model for feature importance?', clf)
start_time = timeit.default_timer()
if mod_feature=='Extra Trees':
    model=etree
    importance=feature_sort(model,X_train,y_train)
elif mod_feature=='Random Forest':
    model=rforest
    importance=feature_sort(model,X_train,y_train)
elapsed = timeit.default_timer() - start_time
st.write('Execution Time for feature selection: %.2f minutes'%(elapsed/60))
#Plot of feature importance
if st.sidebar.checkbox('Show plot of feature importance'):
    plt.bar([x for x in range(len(importance))], importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature (Variable Number)')
    plt.ylabel('Importance')
    st.pyplot()


# In[29]:


#Import performance metrics, imbalanced rectifiers
from sklearn.metrics import  confusion_matrix,classification_report,matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
np.random.seed(42) #for reproducibility since SMOTE and Near Miss use randomizations
smt = SMOTE()
nr = NearMiss()
def compute_performance(model, X_train, y_train,X_test,y_test):
    start_time = timeit.default_timer()
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    'Accuracy: ',scores
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    'Confusion Matrix: ',cm  
    cr=classification_report(y_test, y_pred)
    'Classification Report: ',cr
    mcc= matthews_corrcoef(y_test, y_pred)
    'Matthews Correlation Coefficient: ',mcc
    elapsed = timeit.default_timer() - start_time
    'Execution Time for performance computation: %.2f minutes'%(elapsed/60)
#Run different classification models with rectifiers
if st.sidebar.checkbox('Run a financial fraud detection model'):
    
    alg=['Extra Trees','Random Forest','k Nearest Neighbor','Support Vector Machine','Logistic Regression']
    classifier = st.sidebar.selectbox('Which algorithm?', alg)
    rectifier=['SMOTE','Near Miss','No Rectifier']
    imb_rect = st.sidebar.selectbox('Which imbalanced class rectifier?', rectifier) 
    
    if classifier == 'Extra Trees':
        model=etree
        if imb_rect=='No Rectifier':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)


# In[ ]:




