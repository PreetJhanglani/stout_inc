#!/usr/bin/env python
# coding: utf-8

# ### 1) Importing required packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("casestudy.csv", index_col=0)
data


# In[3]:


data.customer_email.value_counts()


# ## For 2015

# In[4]:


data.year.value_counts()


# In[5]:


data_15 = data[data.year == 2015]
data_15


# ### Total revenue for current year

# In[6]:


print("The Total revenue for 2015 is {}".format(data_15.net_revenue.sum()))


# #### New Customer Revenue

# ##### As we dont have data for 2014 we cant say anything about new customers in 2015.

# #### Existing customers growth

# ##### As we dont have data for 2014 we cant say anything about existing customers in 2015.

# #### Revenue lost from attrition

# ##### As we dont have data for 2014 we cant say anything about new customers in 2015.

# #### Existing Customer Revenue Current Year

# ##### As we dont have data for 2014 we cant say anything about existing customers in 2015.

# #### Existing Customer Revenue Prior Year

# ##### As we dont have data for 2014 we cant say anything about existing customers in 2015.

# ### Total Customers Current Year

# In[7]:


print("The total number of unique customers for 2015 are: {}".format(len(pd.unique(data_15.customer_email))))


# #### Total Customers previous Year

# ##### As we dont have data for 2014 we cant say anything about existing customers in 2015.

# #### New Customers

# ##### As we dont have data for 2014 we cant say anything about existing customers in 2015.

# #### Lost Customers

# ##### As we dont have data for 2014 we cant say anything about existing customers in 2015.

# In[ ]:





# In[ ]:





# ## For 2016

# In[8]:


data_16 = data[data.year == 2016]
data_16


# ### Total revenue for current year

# In[9]:


print("The Total revenue for 2016 is {}".format(data_16.net_revenue.sum()))


# ### New Customer Revenue

# In[10]:


print("The revenue from new customers in 2016 is {}".format(data_16[~data_16.customer_email.isin(data_15.customer_email)].net_revenue.sum()))


# #### Existing customers growth

# In[113]:


exis_16 = list(data_16[data_16.customer_email.isin(data_15.customer_email)].customer_email.values)
exist_gowth = data_16[data_16.customer_email.isin(exis_16)].net_revenue.sum() - data_15[data_15.customer_email.isin(exis_16)].net_revenue.sum()
print("The existing customer growth in 2016 is {}".format(exist_gowth))


# #### Revenue lost from attrition

# In[11]:


print("The revenue lost from attrition in 2016 is {}".format(data_15.net_revenue.sum() - data_16.net_revenue.sum()))


# #### Existing Customer Revenue Current Year

# In[12]:


print("The revenue from existing customers in 2016 is {}".format(data_16[data_16.customer_email.isin(data_15.customer_email)].net_revenue.sum()))


# #### Existing Customer Revenue Prior Year

# In[115]:


print("The revenue from existing customers in 2015 is {}".format(data_15[data_15.customer_email.isin(data_16.customer_email)].net_revenue.sum()))


# ### Total Customers Current Year

# In[14]:


print("The total number of unique customers for 2016 are: {}".format(len(pd.unique(data_16.customer_email))))


# #### Total Customers previous Year

# In[15]:


print("The total number of unique customers for 2015 are: {}".format(len(pd.unique(data_15.customer_email))))


# #### New Customers

# In[117]:


data_16[~data_16.customer_email.isin(data_15.customer_email)]


# In[116]:


len(pd.unique(data_16[~data_16.customer_email.isin(data_15.customer_email)].customer_email))


# #### Lost Customers

# In[17]:


data_15[~data_15.customer_email.isin(data_16.customer_email)]


# In[118]:


len(pd.unique(data_15[~data_15.customer_email.isin(data_16.customer_email)].customer_email))


# In[ ]:





# ## For 2017

# In[18]:


data_17 = data[data.year == 2017]
data_17


# ### Total revenue for current year

# In[19]:


print("The Total revenue for 2017 is {}".format(data_17.net_revenue.sum()))


# ### New Customer Revenue

# In[20]:


print("The revenue from new customers in 2017 is {}".format(data_17[~data_17.customer_email.isin(data_16.customer_email)].net_revenue.sum()))


# #### Existing customers growth

# In[21]:


print("The existing customer growth for 2017 is {}".format(data_17[data_17.customer_email.isin(data_16.customer_email)].net_revenue.sum() - data_16[data_16.customer_email.isin(data_15.customer_email)].net_revenue.sum()))


# #### Revenue lost from attrition

# In[22]:


print("The revenue lost from attrition in 2017 is {}".format(data_16.net_revenue.sum() - data_17.net_revenue.sum()))


# #### Existing Customer Revenue Current Year

# In[23]:


print("The revenue from existing customers in 2017 is {}".format(data_17[data_17.customer_email.isin(data_16.customer_email)].net_revenue.sum()))


# #### Existing Customer Revenue Prior Year

# In[24]:


print("The revenue from existing customers in 2016 is {}".format(data_16[data_16.customer_email.isin(data_17.customer_email)].net_revenue.sum()))


# ### Total Customers Current Year

# In[25]:


print("The total number of unique customers for 2017 are: {}".format(len(pd.unique(data_17.customer_email))))


# #### Total Customers previous Year

# In[26]:


print("The total number of unique customers for 2016 are: {}".format(len(pd.unique(data_16.customer_email))))


# #### New Customers

# In[27]:


data_17[~data_17.customer_email.isin(data_16.customer_email)]


# #### Lost Customers

# In[28]:


data_16[~data_16.customer_email.isin(data_17.customer_email)]


# In[ ]:





# In[ ]:





# ## Visualization

# In[33]:


data.groupby(by='year').sum().reset_index()['net_revenue']/1000000


# In[45]:


plt.figure(figsize = (10,10))
ax = sns.barplot(data = data.groupby(by='year').sum().reset_index(), x = data.groupby(by='year').sum().reset_index()['year'], y=data.groupby(by='year').sum().reset_index()['net_revenue']/1000000,  palette="Blues_d")
ax.bar_label(ax.containers[0])
ax.set_ylabel("net_revenue * 10^6")
ax.figure.savefig("net_revenue.jpg")


# In[ ]:





# #### 2016 new v/s existing

# In[43]:


new_exi = {"New":data_16[~data_16.customer_email.isin(data_15.customer_email)].net_revenue.sum()/1000000, "Existing":data_16[data_16.customer_email.isin(data_15.customer_email)].net_revenue.sum()/1000000}
new_exi


# In[47]:


vals = [i for i in new_exi.values()]
plt.figure(figsize = (10,10))
ax = sns.barplot(x = list(new_exi.keys()), y=vals)
ax.set_title("2016: New v/s Existing Customers")
ax.bar_label(ax.containers[0])
ax.set_ylabel("net_revenue * 10^6")
ax.figure.savefig("16_new_exis.jpg")


# In[ ]:





# In[ ]:





# #### 2017 new v/s existing

# In[48]:


new_exi = {"New":data_17[~data_17.customer_email.isin(data_16.customer_email)].net_revenue.sum()/1000000, "Existing":data_17[data_17.customer_email.isin(data_16.customer_email)].net_revenue.sum()/1000000}
new_exi


# In[49]:


vals = [i for i in new_exi.values()]
plt.figure(figsize = (10,10))
ax = sns.barplot(x = list(new_exi.keys()), y=vals)
ax.set_title("2017: New v/s Existing Customers")
ax.bar_label(ax.containers[0])
ax.set_ylabel("net_revenue * 10^6")
ax.figure.savefig("17_new_exis.jpg")


# ##### As we can see the revenue obtained from existing customers is pretty less in 2017 as compared to 2016.

# In[ ]:





# In[59]:


data_15.groupby("customer_email").sum().reset_index().sort_values(by="net_revenue", ascending=False).net_revenue.mean()


# In[60]:


data_16.groupby("customer_email").sum().reset_index().sort_values(by="net_revenue", ascending=False).net_revenue.mean()


# In[61]:


data_17.groupby("customer_email").sum().reset_index().sort_values(by="net_revenue", ascending=False).net_revenue.mean()


# In[62]:


mean_yr = {'2015':data_15.groupby("customer_email").sum().reset_index().sort_values(by="net_revenue", ascending=False).net_revenue.mean(),
          '2016':data_16.groupby("customer_email").sum().reset_index().sort_values(by="net_revenue", ascending=False).net_revenue.mean(),
          '2017':data_17.groupby("customer_email").sum().reset_index().sort_values(by="net_revenue", ascending=False).net_revenue.mean()}
mean_yr


# In[69]:


vals = [i for i in mean_yr.values()]
plt.figure(figsize = (10,10))
ax = sns.barplot(x = list(mean_yr.keys()), y=vals, palette='Purples')
ax.set_title("Year v/s Mean revenue")
ax.bar_label(ax.containers[0])
ax.set_ylabel("mean_revenue")
ax.figure.savefig("mean_yr.jpg")


# In[79]:


for i, j in mean_yr.items():
    print(i,j)


# In[84]:


vals = [i for i in mean_yr.values()]
plt.figure(figsize = (10,10))
ax = sns.scatterplot(x = list(mean_yr.keys()), y=vals, palette='Purples', s=100)
for i, j in mean_yr.items():
    plt.text(i,j+0.005,str(j))
ax.set_title("Year v/s Mean revenue")
ax.set_ylabel("mean_revenue")
ax.figure.savefig("mean_yr.jpg")


# In[ ]:





# In[88]:


plt.figure(figsize = (10,10))
ax = sns.boxplot(y=data.net_revenue, x = data.year, palette='RdBu')


# In[ ]:





# In[94]:


exis_16 = list(data_16[data_16.customer_email.isin(data_15.customer_email)].customer_email.values)
exis_16


# In[97]:


exis_17 = list(data_17[data_17.customer_email.isin(exis_16)].customer_email.values)
exis_17


# In[100]:


data_17[data_17.customer_email.isin(exis_16)].net_revenue.sum()


# In[102]:


data_16[data_16.customer_email.isin(exis_17)].net_revenue.sum()


# In[104]:


data_15[data_15.customer_email.isin(exis_17)].net_revenue.sum()


# In[110]:


exist = {'2015':data_15[data_15.customer_email.isin(exis_17)].net_revenue.sum(),
        '2016':data_16[data_16.customer_email.isin(exis_17)].net_revenue.sum(),
        '2017':data_17[data_17.customer_email.isin(exis_16)].net_revenue.sum()}
vals = [i for i in exist.values()]
plt.figure(figsize = (10,10))
ax = sns.lineplot(x = list(exist.keys()), y=vals, palette='Purples', marker = "o")
for i, j in exist.items():
    plt.text(i,j+0.005,str(j))
ax.set_title("Existing Customer Revenue over the year")
ax.set_ylabel("total_revenue")
ax.figure.savefig("exist_yr.jpg")


# In[ ]:




