
# coding: utf-8

# In[177]:


import warnings                  
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')

import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

pd.options.display.max_rows = 100


# In[178]:


data = pd.read_csv('C:/Users/Win_10/Desktop/train.csv')#read train and display
data.head()


# In[252]:


data.describe()


# In[179]:


data = data.drop(['Ticket'], axis=1)#drop unnecessary feature


# In[180]:


data.head()


# In[182]:



data['Age'].fillna(data['Age'].median(), inplace=True)# replace the null values with the median age 
data.describe()#descirbe numerical features


# In[183]:


survived_sex = data[data['Survived']==1]['Sex'].value_counts()#correlation between survived and gender
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar')


# In[184]:


figure = plt.figure(figsize=(10,8))#correlation between survived and age
plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], stacked=True, color = ['b','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')


# In[185]:


figure = plt.figure(figsize=(10,8))#correlation between survived and fare
plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']], stacked=True, color = ['b','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')


# In[186]:


survived_embark = data[data['Survived']==1]['Embarked'].value_counts()#correlation between survived and embarkation
dead_embark = data[data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar')


# In[187]:


plt.figure(figsize=(20,8))#correlation between gender and age

ax = plt.subplot()
ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Sex'],c='blue')
ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Sex'],c='red')

ax.set_xlabel('Age')
ax.set_ylabel('Sex')



# In[188]:


plt.figure(figsize=(20,8))#correlation between fare and age

ax = plt.subplot()
ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Fare'],c='blue')
ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Fare'],c='red')

ax.set_xlabel('Age')
ax.set_ylabel('Fare')


# In[189]:


plt.figure(figsize=(20,8))#correlation between fare and pclass

ax = plt.subplot()
ax.scatter(data[data['Survived']==1]['Pclass'],data[data['Survived']==1]['Fare'],c='blue')
ax.scatter(data[data['Survived']==0]['Pclass'],data[data['Survived']==0]['Fare'],c='red')

ax.set_xlabel('Pclass')
ax.set_ylabel('Fare')


# In[190]:


def get_combined_data():
    train = pd.read_csv('C:/Users/Win_10/Desktop/train.csv')
    test = pd.read_csv('C:/Users/Win_10/Desktop/test.csv')
    targets = train.Survived  #Seprate survived feature then drop it
    train.drop('Survived', 1, inplace=True)
    
    combined = train.append(test) # merging train data and test data 
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    
    return combined


# In[191]:


combined = get_combined_data()


# In[192]:


combined.head()


# In[193]:


def process_names():
    
    global combined
    #drop names
    combined.drop('Name',axis=1,inplace=True)

    


# In[194]:


process_names()


# In[195]:


combined.head()


# In[196]:


grouped_train = combined.head(891).groupby(['Sex','Pclass'])# group dataset and compute the nedian age
grouped_median_train = grouped_train.median()

grouped_test = combined.iloc[891:].groupby(['Sex','Pclass'])
grouped_median_test = grouped_test.median()


# In[197]:


def process_age():#replace missing age with median age by class and sex
    
    global combined
    def fillAges(row, grouped_median):
        if row['Sex']=='female' and row['Pclass'] == 1:
                return grouped_median.loc['female', 1]['Age']

        elif row['Sex']=='female' and row['Pclass'] == 2:
                return grouped_median.loc['female', 2]['Age']
            
        elif row['Sex']=='female' and row['Pclass'] == 3:
                return grouped_median.loc['female', 3]['Age']

        elif row['Sex']=='male' and row['Pclass'] == 1:
                return grouped_median.loc['male', 1]['Age']

        elif row['Sex']=='male' and row['Pclass'] == 2:
                return grouped_median.loc['male', 2]['Age']

        elif row['Sex']=='male' and row['Pclass'] == 3:
                return grouped_median.loc['male', 3]['Age']
    
    combined.head(891).Age = combined.head(891).apply(lambda r : fillAges(r, grouped_median_train) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)
    
    combined.iloc[891:].Age = combined.iloc[891:].apply(lambda r : fillAges(r, grouped_median_test) if np.isnan(r['Age']) 
                                                      else r['Age'], axis=1)
    



# In[198]:


process_age()


# In[199]:


combined.head()


# In[200]:



def process_cabin(): # clean the cabin
    
    global combined
    combined.drop('Cabin',axis=1,inplace=True)


# In[201]:


process_cabin()


# In[202]:


combined.head()


# In[203]:



def process_ticket(): # clean the ticket
    
    global combined
    combined.drop('Ticket',axis=1,inplace=True)


# In[204]:


process_ticket()


# In[205]:


combined.head()


# In[206]:


def process_embarked():
    #replace missing values wuth most frequent onw
    
    global combined
    combined.head(891).Embarked.fillna('S', inplace=True)
    combined.iloc[891:].Embarked.fillna('S', inplace=True)
    
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')#dummy encoding 
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)


# In[207]:


process_embarked()


# In[208]:


def process_fares():
    #replace missing value with mean
    global combined
    combined.head(891).Fare.fillna(combined.head(891).Fare.mean(), inplace=True)
    combined.iloc[891:].Fare.fillna(combined.iloc[891:].Fare.mean(), inplace=True)


# In[209]:


process_fares()


# combined.head()

# In[210]:


combined.head()


# In[211]:


def process_pclass():#dummy encoding on pclass( 3 class : 1,2,3 )
    
    global combined
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    combined = pd.concat([combined,pclass_dummies],axis=1)    
    combined.drop('Pclass',axis=1,inplace=True)
    


# In[212]:


process_pclass()


# In[213]:


combined.head()


# In[214]:



def process_sex():   #convert gender to numeric
    global combined
    
    combined['Sex'] = combined['Sex'].map({'female':1,'male':0})
    


# In[215]:


process_sex()


# In[216]:


combined.head()


# In[217]:



def process_passengerID(): # clean the ticket
    
    global combined
    combined.drop('PassengerId',axis=1,inplace=True)


# In[218]:


process_passengerID()


# In[219]:


combined.head()


# In[220]:


def process_family():
    #add 4 features related to familysize instead of Parch and SibSp
    global combined
    
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5<=s else 0)


# In[221]:


process_family()


# In[222]:


combined.head()


# In[296]:


from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold


# In[297]:


def seprate_combined():
    global combined
    
    train0 = pd.read_csv('C:/Users/Win_10/Desktop/train.csv')
    targets = train0.Survived
    train = combined.head(891)
    test = combined.iloc[891:]
    
    return train, test, targets



# In[298]:


seprate_combined()


# In[299]:


train, test, targets = seprate_combined()


# In[300]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)


# In[301]:




# turn run_gs to True if you want to run the gridsearch again.
run_gs = False

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [1, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(targets, n_folds=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(train, targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
    model = RandomForestClassifier(**parameters)
    model.fit(train, targets)



# In[302]:


def compute_score(clf, X, y, scoring='accuracy'):#compute accuracy rate
    score = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(score)


# In[303]:



compute_score(model, train, targets, scoring='accuracy')


# In[304]:


output = model.predict(test).astype(int)#wite output into file
df_output = pd.DataFrame()
aux = pd.read_csv('C:/Users/Win_10/Desktop/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('C:/Users/Win_10/Desktop/output.csv',index=False)

