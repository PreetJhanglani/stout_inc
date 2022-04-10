#!/usr/bin/env python
# coding: utf-8

# ## Name: Preet Jhanglani
# ### Case Study 1
# ### Lending club dataset

# ![image.png](attachment:image.png)
# 
# **A Brief about LendingClub: **
# 
# **LendingClub** is the first US based peer to peer lending company, headquarter in SAN Francisco, California to register its offerings as securities and exchange commission. It offers loan trading on secondary market. LendingClub enables borrowers to create unsecured personal loans between 1000 and 40000 with standard loan period of 3 years. LendingClub acts like the "bridge" between borrowers and Investors.
# 
# 

# ![image.png](attachment:image.png)
# 
# **Why do they need this analysis?**
# 
# From above working model, it is clear that its very important for LendingClub to know if there is any chance of their borrowers defaulting.

# ### 1) Importing the required libraries

# In[54]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.io as pio
pio.renderers.default='notebook'
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
import kaleido
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2) Reading the Data

# In[3]:


data = pd.read_csv("loans_full_schema.csv")
data


# In[4]:


df = deepcopy(data)


# #### Lets take a look at dataset columns

# In[5]:


data.columns


# ##### There are 55 columns and we need to use the most correlated data columns to predict the interest rate

# ### 3) Pre Processing

# In[6]:


data.describe()


# ##### From the above describe data we can see the count of all the columns is not equal to the length of data.That implies the data has missing values.

# In[7]:


data.isnull().sum()


# In[8]:


print(data.isnull().any().value_counts(), "\n")
print(f"The columns that have missing values are total {data.isnull().any().sum()}")


# In[9]:


null_columns = data.columns[data.isnull().sum() > 0].tolist()
null_columns


# ##### Lets look at these columns with missing values and if the count of missing values is greater than 50%, we will drop that column

# In[10]:


drop_cols = data.loc[:,null_columns].isnull().sum()[data.loc[:,null_columns].isnull().sum().values > 5000].index.tolist()
drop_cols


# ##### Lets drop the columns mentioned above

# In[11]:


data.drop(columns=drop_cols, axis=1, inplace = True)
null_columns = data.columns[data.isnull().sum() > 0].tolist()
null_columns


# In[12]:


null_columns


# ##### For the above columns with null lets take a look and try to replace null with some value

# In[13]:


data["emp_title"].value_counts().index[0]


# The emp_title column has mode as manager. Lets replace the NaNs with manager

# In[14]:


data['emp_title'].fillna(data["emp_title"].value_counts().index[0], inplace = True)
data['emp_title'].isna().sum()


# ##### Lets look at emp_length column

# In[15]:


data['emp_length']


# In[16]:


data.emp_length.describe()


# In[17]:


data.emp_length.value_counts()


# The mode for the emp_length column is 10.0 and the mean is 5.930306. We will round this mean to 6.0 and replace NA with this mean

# In[18]:


data.emp_length.fillna(round(data.emp_length.mean()), inplace = True)
data.emp_length.isna().sum()


# ##### Now debt_to_income column

# In[19]:


data.debt_to_income.describe()


# In[20]:


data.debt_to_income.value_counts()


# Replacing the NA with mean i.e 19.308192

# In[21]:


data.debt_to_income.fillna(data.debt_to_income.mean(),inplace = True)
data.debt_to_income.isna().sum()


# In[22]:


null_columns


# ##### Now months_since_last_credit_inquiry columns need to be examined

# In[23]:


data.months_since_last_credit_inquiry


# In[24]:


data.months_since_last_credit_inquiry.describe()


# In[25]:


data.months_since_last_credit_inquiry.value_counts()


# Again replace na with mean

# In[26]:


data.months_since_last_credit_inquiry.fillna(round(data.months_since_last_credit_inquiry.mean()), inplace = True)
data.months_since_last_credit_inquiry.isna().sum()


# ##### Now num_accounts_120d_past_due to be examined

# In[27]:


data.num_accounts_120d_past_due


# In[28]:


data.num_accounts_120d_past_due.describe()


# As all the values in this columns are 0 we can drop this column

# In[29]:


data.drop('num_accounts_120d_past_due', inplace = True, axis = 1)
data


# #### As seen in the num_accounts_120d_past_due column. A column can consists of all 0. Lets look at the numeric columns

# In[30]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
data_num = data.select_dtypes(include=numerics)


# In[31]:


data_num


# In[32]:


data_num.columns[(data_num == 0).all()]


# #### As we can see there are no columns with all zero values but there might be some columns with majority of zeros

# In[33]:


data_num.paid_late_fees.value_counts()


# In[34]:


zeroes = (data_num[data_num == 0].count(axis=0)/len(data_num.index))
zeroes[zeroes > 0.9]


# The above columns have zeroes more than 90% of the total values. We can remove these columns

# In[35]:


data.drop(columns= zeroes[zeroes > 0.9].index.tolist(), axis = 1, inplace=True)
data


# In[36]:


data.interest_rate


# #### Convert the string object data columns to numerical categories

# In[37]:


data.info()


# ##### We will factorise the object datatypes columns

# In[38]:


obj_cols = data.select_dtypes(include=[object]).columns.tolist()
obj_cols


# Except issue_month we will convert all object columns to factors

# In[39]:


obj_cols.pop(obj_cols.index('issue_month'))
obj_cols


# In[40]:


df_copy = deepcopy(data)


# In[186]:


le = LabelEncoder()
for i in obj_cols:
    data.loc[:,i] = le.fit_transform(data.loc[:,i]) 


# In[187]:


data


# In[188]:


data.to_csv('data_clean.csv', index=False)


# In[ ]:





# In[ ]:





# ### 3) EDA

# In[189]:


data_num = data.select_dtypes(include = numerics)


# In[190]:


data.corr()['interest_rate'].sort_values()


# #### Lets plot the above correlation values of interest rates into a heatmap

# In[191]:


plt.figure(figsize=(10,10))
heat = sns.heatmap(pd.DataFrame(data.corr()['interest_rate']), cmap = 'YlGnBu', annot=True, linewidths=.5)
heat.figure.savefig('heatmap.jpg')


# From the above heatmap we can say the columns of grade and sub_grade are most correlated with interest_rate. The above correlation isnt with issues_month as its a string data and I feel the interest rate should be correlated with that too.

# ##### Interest rate v/s issue_month

# In[192]:


data['issue_month'].value_counts()


# In[205]:


issue = data[['issue_month','interest_rate']].groupby('issue_month').mean().reset_index()
issue['issue_month_num'] = pd.to_datetime(issue.issue_month)


# In[207]:


issue.sort_values(by = 'issue_month_num', inplace = True)
issue


# In[52]:


data.interest_rate.value_counts()


# In[210]:


plt.figure(figsize=(10,10))
ax = sns.barplot(data = issue, x = 'issue_month', y = 'interest_rate')
ax.bar_label(ax.containers[0])
ax.figure.savefig('issue_month.jpg')


# ##### From the above bar plot we can say that the month of Feb in 2018 has a higher mean interest rate

# In[107]:


plt.figure(figsize=(10,10))
dist = sns.distplot(np.log(data.interest_rate))
dist.figure.savefig('dist.jpg')


# ##### From the above distribution plot we can say that the interest rate is not normally distributed.

# In[55]:


data.grade.value_counts()


# In[57]:


df_copy.grade


# In[59]:


df_copy.interest_rate


# In[61]:


df_copy[['grade','interest_rate']].groupby('grade').mean()


# In[76]:


df_copy[['sub_grade','interest_rate']].groupby('sub_grade').mean().reset_index()


# In[108]:


plt.figure(figsize=(10,10))
ax = sns.barplot(data = df_copy[['grade','sub_grade','interest_rate']].groupby('grade').mean().reset_index(), x = 'grade', y = 'interest_rate')
ax.bar_label(ax.containers[0])
ax.figure.savefig('grade.jpg')


# ##### In the above graph its clearly depicted that the interest rate is lower if the grade is higher.

# In[98]:


int_rate = dict(df_copy[['emp_length','interest_rate']].groupby('emp_length').mean().reset_index()['interest_rate'])


# In[110]:


plt.figure(figsize=(10,10))
ax = sns.lineplot(data = df_copy[['emp_length','interest_rate']].groupby('emp_length').mean().reset_index(), x = 'emp_length', y='interest_rate', marker = 'o')
for i in int_rate:
    plt.text(i,int_rate[i] + 0.005,str(int_rate[i]))
ax.figure.savefig('emp_length.jpg')


# In[ ]:





# In[44]:


states = df_copy[['state','interest_rate']].groupby(by='state').mean()
states.reset_index(inplace=True)
states


# In[59]:


fig = px.choropleth(states,
                    locations='state', 
                    locationmode="USA-states", 
                    scope="usa",
                    color='interest_rate',
                    color_continuous_scale="Viridis_r",labels = dict(zip(states.state, states.interest_rate)))
# plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')
plotly.offline.plot(fig)


# In[ ]:





# In[ ]:





# ### 4) Modeling 

# #### For better accuracy we can scale the columns and we can even select the best features using PCA. We dont have that much time so we will implement a basic linear regression and a basic neural network without cross validation.

# In[130]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense,Dropout
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression


# In[124]:


le = LabelEncoder()
data['issue_month'] = le.fit_transform(data.issue_month)


# In[125]:


y = data['interest_rate']
x = data.drop('interest_rate', axis = 1)


# In[126]:


x.shape


# In[127]:


y.shape


# In[128]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[129]:


lr = LinearRegression()
lr.fit(X_train,y_train)


# In[131]:


#predict the interest rate
predictions = lr.predict(X_test)


# In[132]:


r2_score(y_test,predictions)


# In[134]:


print("Accuracy : {}".format(100 - mean_squared_error(y_test,predictions)))


# #### Using Linear Regression we were able to achieve an accuracy of 99.72%. 

# #### Now lets implement Neural Network

# In[140]:


model = Sequential()
model.add(Dense(86, input_dim=43, activation= "relu"))
model.add(Dense(43, activation= "relu"))
model.add(Dense(20, activation= "relu"))
model.add(Dense(1, activation= "relu"))
model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
model.fit(X_train, y_train,batch_size=50, epochs=100,verbose=1)
pred= model.predict(X_test)


# In[139]:


pred


# In[141]:


from sklearn.svm import SVR
regrassor = SVR(kernel = 'rbf')
regrassor.fit(X_train, y_train)


# In[142]:


y_pred = regrassor.predict(X_test)


# In[143]:


mean_squared_error(y_test,y_pred)


# In[144]:


print("Accuracy : {}".format(100 - mean_squared_error(y_test,y_pred)))


# #### We can implement more algorithms and even hypertune these but it requires time.

# In[ ]:




