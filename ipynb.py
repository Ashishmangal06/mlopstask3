#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('ggplot')


# In[79]:


dataset = pd.read_csv("placement_train.csv")
dataset.head()


# In[80]:


for col in dataset.columns.unique():
    print('\n', col ,'\n', dataset[col].unique())


# In[81]:


dataset1=dataset.copy()


# In[82]:


dataset.isna().any()


# In[83]:


dataset.info()


# In[84]:


fig, axs = plt.subplots(ncols=4,figsize=(20,5))
sns.countplot(dataset['gender'], ax = axs[0])
sns.countplot(dataset['ssc_b'], ax = axs[1], palette="vlag")
sns.countplot(dataset['hsc_b'], ax = axs[2], palette="rocket")
sns.countplot(dataset['hsc_s'], ax = axs[3], palette="deep")


# In[85]:


fig, axs = plt.subplots(ncols=3,figsize=(20,5))
sns.countplot(dataset['workex'], ax = axs[0], palette="Paired")
sns.countplot(dataset['specialisation'], ax = axs[1], palette="muted")
sns.countplot(dataset['status'], ax = axs[2],palette="dark")


# In[86]:


dataset = dataset.drop(['sl_no'], axis = 1)


# In[87]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[88]:


dataset['gender'] = le.fit_transform(dataset['gender'])
dataset['ssc_b'] = le.fit_transform(dataset['ssc_b'])
dataset['workex'] = le.fit_transform(dataset['workex'])
dataset['specialisation'] = le.fit_transform(dataset['specialisation'])
dataset['status'] = le.fit_transform(dataset['status'])
dataset['hsc_b'] = le.fit_transform(dataset['hsc_b'])
dataset['hsc_s'] = le.fit_transform(dataset['hsc_s'])
dataset['degree_t'] = le.fit_transform(dataset['degree_t'])


# In[89]:


plt.figure(figsize=(15,10))
corr = dataset.corr()
sns.heatmap(corr, annot = True)


# In[90]:


plt.figure(figsize=(5,5))
sns.scatterplot(x='status', y = 'degree_p', hue ='gender', data = dataset1)


# In[91]:


plt.figure(figsize=(7,7))
plt.hist(dataset1['salary'], bins = 10)
plt.show()


# In[92]:


fig, axs = plt.subplots(ncols=3,figsize=(20,5))
sns.scatterplot(x = 'degree_p',y='hsc_p',hue='status',data = dataset1, ax= axs[0])
sns.scatterplot(x = 'degree_p',y='hsc_p',hue='gender',data = dataset1, ax= axs[1], palette="muted")
sns.scatterplot(x = 'degree_p',y='hsc_p',hue='degree_t',data = dataset1, palette="dark", ax= axs[2])


# In[93]:


fig, axs = plt.subplots(ncols=3,figsize=(20,5))
sns.scatterplot(x = 'degree_p',y='hsc_p',hue='hsc_s',data = dataset1, ax= axs[0])
sns.scatterplot(x = 'degree_p',hue='specialisation',y='mba_p',data = dataset1, ax= axs[1], palette="muted")
sns.scatterplot(x = 'degree_p',hue='workex',y='salary',data = dataset1, palette="dark", ax= axs[2])


# In[94]:


dataset = pd.DataFrame(dataset)


# In[95]:


dataset_placed = dataset1[dataset1['status'] == 'Placed']


# In[96]:


dataset2 = dataset_placed.groupby(['degree_t','degree_p','mba_p','specialisation','hsc_p', 'hsc_s','salary', 'workex']).sum().sort_values(by ='salary')
dataset2


# In[97]:


dataset_np = dataset1[dataset1.status == 'Not Placed']


# In[98]:


dataset3 = dataset_np.groupby(['degree_t','degree_p','mba_p','specialisation','hsc_p', 'hsc_s','workex']).sum().sort_values(by ='degree_p')
dataset3


# In[99]:


X = dataset.drop(['status'], axis = 1)
y = dataset['status']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 43)


# In[100]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[102]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8)
classifier.fit( X_train, y_train)


# In[103]:


y_pred = classifier.predict(X_test)


# In[104]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


# In[ ]:




