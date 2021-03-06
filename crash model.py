
# coding: utf-8

# In[2]:


# importing libraries
import pandas as pd
import numpy as np


# In[3]:


# reading the data
df = pd.read_csv('NSE-TATAGLOBAL11.csv')


# In[4]:


# looking at the first five rows of the data
print(df.head())
print('\n Shape of the data:')
print(df.shape)


# In[9]:


# setting the index as date
df.set_index(df['Date'], drop = False , append = False, inplace = False, verify_integrity = False).drop('Date',1)


# In[10]:



#creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]


# In[11]:



# NOTE: While splitting the data into train and validation set, we cannot use random splitting since that will destroy the time component. So here we have set the last year’s data into validation and the 4 years’ data before that into train set.

# splitting into train and validation
train = new_data[:987]
valid = new_data[987:]


# In[12]:



# shapes of training set
print('\n Shape of training set:')
print(train.shape)


# In[13]:



# shapes of validation set
print('\n Shape of validation set:')
print(valid.shape)


# In[14]:



# In the next step, we will create predictions for the validation set and check the RMSE using the actual values.
# making predictions
preds = []
for i in range(0,valid.shape[0]):
    a = train['Close'][len(train)-248+i:].sum() + sum(preds)
    b = a/248
    preds.append(b)


# In[15]:



# checking the results (RMSE value)
rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-preds),2)))
print('\n RMSE value on validation set:')
print(rms)

