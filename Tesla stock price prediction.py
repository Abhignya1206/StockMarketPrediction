#!/usr/bin/env python
# coding: utf-8

# In[6]:


#RA2111027010022(Anusha Patra)
#RA2111027010067(Abhignya Priyadarshini)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

#for offline plotting
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 


# In[7]:


tesla = pd.read_csv(r"C:\Users\anush\OneDrive\Desktop\ML Project\tesla.csv")
tesla.head()


# In[8]:


tesla.info()


# In[11]:


tesla['Date'] = pd.to_datetime(tesla['Date'], format='%d-%m-%Y')

# Now you can calculate the minimum and maximum dates and the total number of days
min_date = tesla['Date'].min()
max_date = tesla['Date'].max()
total_days = (max_date - min_date).days

print(f'Dataframe contains stock prices between {min_date} and {max_date}')
print(f'Total days = {total_days} days')


# In[12]:


tesla.describe()


# In[13]:


tesla[['Open','High','Low','Close','Adj Close']].plot(kind='box')


# In[14]:


# Setting the layout for our plot
layout = go.Layout(
    title='Stock Prices of Tesla',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Price',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

tesla_data = [{'x':tesla['Date'], 'y':tesla['Close']}]
plot = go.Figure(data=tesla_data, layout=layout)


# In[15]:


#plot(plot) #plotting offline
iplot(plot)


# In[16]:


# Building the regression model
from sklearn.model_selection import train_test_split

#For preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#For model evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[17]:


#Split the data into train and test sets
X = np.array(tesla.index).reshape(-1,1)
Y = tesla['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)


# In[18]:


# Feature scaling
scaler = StandardScaler().fit(X_train)


# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


#Creating a linear model
lm = LinearRegression()
lm.fit(X_train, Y_train)


# In[21]:


#Plot actual and predicted values for train dataset
trace0 = go.Scatter(
    x = X_train.T[0],
    y = Y_train,
    mode = 'markers',
    name = 'Actual'
)
trace1 = go.Scatter(
    x = X_train.T[0],
    y = lm.predict(X_train).T,
    mode = 'lines',
    name = 'Predicted'
)
tesla_data = [trace0,trace1]
layout.xaxis.title.text = 'Day'
plot2 = go.Figure(data=tesla_data, layout=layout)


# In[22]:


iplot(plot2)


# In[23]:


#Calculate scores for model evaluation
scores = f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t{r2_score(Y_test, lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t{mse(Y_test, lm.predict(X_test))}
'''
print(scores)


# In[ ]:




