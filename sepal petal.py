
# coding: utf-8

# In[23]:


import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[31]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
dataset=pandas.read_csv(url, names=names)


# In[32]:


print(dataset.shape)


# In[35]:


print(dataset.head(30))


# In[49]:


print(dataset.describe())


# In[57]:


dataset.plot(kind='box', subplots = True, layout = (4,4) , sharex = False, sharey = False)
plt.show()


# In[58]:


dataset.hist()
plt.show()


# In[59]:


scatter_matrix(dataset)
plt.show()


# In[65]:


array = dataset.values
x = array[:,0:3]
y = array[:,3]
validation_size = 0.20
seed = 6
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size = validation_size ,random_state = seed)


# In[66]:


seed=6
scoring = "accuracy"


# In[71]:


# spot check algo
models = []
models.append(("LR", LogisticRegression()))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM",SVC()))
results = []
names = []
for  name ,model in models:
    kfold= model_selection.KFold(n_splits=10,random_state =seed)
    cv_results = model_selection.cross_val_score(model,x_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s:%f (%f)}"% (name ,cv_results.mean(),cv_results.std())
    print(msg)

