#!/usr/bin/env python
# coding: utf-8

# In[7]:


#get_ipython().system('pip install scikit-learn')

# In[10]:


import sklearn as sl


# In[11]:


from sklearn.datasets import load_iris


# In[12]:


iris = load_iris()


# In[13]:


type(iris)


# In[14]:


iris.data


# In[15]:


print(iris.feature_names)


# In[16]:


print(iris.target)


# In[17]:


print(iris.target_names)


# In[18]:


print(type(iris.data))
print(type(iris.target))


# In[19]:


print(iris.data.shape)


# In[25]:


from sklearn.model_selection import train_test_split


# In[27]:


X = iris.data
y = iris.target


# In[28]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)


# In[29]:


print(X_train.shape)
print(X_test.shape)


# In[30]:


print(y_train.shape)
print(y_test.shape)


# In[31]:


from sklearn.neighbors import KNeighborsClassifier


# In[32]:


from sklearn import metrics


# In[36]:


k_range = range(1,26)
scores = {}
scores_list = []


# In[37]:


for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))


# In[46]:


# get_ipython().system('pip install matplotlib')


# In[47]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot(k_range,scores_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# In[48]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)


# In[49]:


classes = {0:'setosa',1:'versicolor',2:'virginica'}

x_new = [[3,4,5,2],[5,4,2,2]]
y_predict = knn.predict(x_new)

print(classes[y_predict[0]])
print(classes[y_predict[1]])


# In[50]:


# get_ipython().system('pip install streamlit')


# In[51]:


import streamlit as st


# In[59]:


# sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
sepal_length = st.slider("Sepal length (cm)", 0.00, 10.00)
sepal_width = st.slider("Sepal width (cm)", 0.00, 10.00)
petal_length = st.slider("Petal length (cm)", 0.00, 10.00)
petal_width = st.slider("Petal width (cm)", 0.00, 10.00)
x_new = [[sepal_length,sepal_width,petal_length,petal_width]]
y_predict = knn.predict(x_new)

st.write("# Classification")
st.write(classes[y_predict[0]])
