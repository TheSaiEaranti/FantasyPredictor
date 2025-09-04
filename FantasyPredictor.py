#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd


# In[7]:


matches = pd.read_csv('matches.csv', index_col = 0)


# In[8]:


matches.head()


# In[5]:


matches.shape


# In[6]:


38 * 20 * 2


# In[7]:


matches['team'].value_counts()


# In[9]:


matches[matches["team"] == "Liverpool"]


# In[10]:


matches['round'].value_counts()


# In[9]:


matches.dtypes


# In[10]:


matches['date'] = pd.to_datetime(matches['date'])


# In[11]:


matches


# In[12]:


matches.dtypes


# In[13]:


matches["venue_code"] = matches["venue"].astype("category").cat.codes


# In[15]:


matches["opp_code"] = matches["opponent"].astype("category").cat.codes


# In[17]:


matches["hour"] = matches["time"].str.replace(":.+", "", regex = True).astype("int")


# In[19]:


matches["day_code"] = matches["date"].dt.dayofweek


# In[21]:


matches["target"] = (matches["result"] == "W").astype("int")


# In[22]:


matches


# In[23]:


from sklearn.ensemble import RandomForestClassifier


# In[24]:


rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)


# In[26]:


train = matches[matches["date"]<'2022-01-01']


# In[27]:


test = matches[matches["date"]>'2022-01-01']


# In[32]:


predictors = ["venue_code", "opp_code", "hour", "day_code"]


# In[33]:


rf.fit(train[predictors], train["target"])


# In[34]:


preds = rf.predict(test[predictors])


# In[36]:


from sklearn.metrics import accuracy_score


# In[38]:


acc = accuracy_score(test["target"], preds)


# In[39]:


acc


# In[40]:


combined = pd.DataFrame(dict(actual=test["target"], prediction = preds))


# In[41]:


pd.crosstab(index=combined["actual"], columns = combined["prediction"])


# In[42]:


from sklearn.metrics import precision_score


# In[43]:


precision_score(test["target"], preds)


# In[44]:


grouped_matches = matches.groupby("team")


# In[45]:


group = grouped_matches.get_group("Manchester City")


# In[46]:


group


# In[48]:


def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed = 'left').mean()
    group[new_cols] = group.dropna(subset=new_cols)
    return group


# In[49]:


cols = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
new_cols = [f"{c}_rolling" for c in cols]


# In[50]:


new_cols


# In[53]:


rolling_averages(group, cols, new_cols)


# In[ ]:




