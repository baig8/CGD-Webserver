
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('wbdcdata.csv')

#select independent variables
X = df[["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension", "worst_radius", "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness", "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry", "worst_fractal_dimension", "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness", "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry", "mean_fractal_dimension"]]

#dependent variable

y = df["diagnosis"]


# In[69]:


#split the data into independent X and dependent Y datasets
#X = df.iloc[:,2:32].values
#y = df.iloc[:,1]


# In[84]:





# In[70]:


#Split dataset into train and test data

from sklearn.model_selection import train_test_split


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)


# In[72]:


#Feature Scaling

from sklearn.preprocessing import StandardScaler


# In[73]:


scale = StandardScaler()


# In[74]:


X_train = scale.fit_transform(X_train)


# In[75]:


X_test = scale.fit_transform(X_test)


# In[76]:


#y_train = scale.fit_transform(y_train)


# In[77]:


#y_test = scale.fit_transform(y_test)

#Instantiate the models

#using Random forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

#make pickle file
import pickle

#used to fit
pickle.dump(classifier, open("model_new.pkl", "wb"))


