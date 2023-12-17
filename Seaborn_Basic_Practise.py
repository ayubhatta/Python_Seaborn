#!/usr/bin/env python
# coding: utf-8

# # Line Plot In Seaborn

# ''' 
# Seaborn is a Python data visualization library based on matplotlib. 
# It provides a high-level interface for drawing attractive and informative
# statistical graphics.
# '''

# In[1]:

# importing seaborn as sns
import seaborn as sns


# In[4]:

# importing numpy, pandas and matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


roll_no = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
marks = [23,45,67,89,56,34,21,45,67,32,67,76,33,21,45]
sample_df = pd.DataFrame({"Rollno": roll_no, "Marks": marks})
sample_df.head()


# # Line Plot

# In[6]:


sns.lineplot(x='Rollno', y='Marks', data=sample_df)
plt.title('Student Marks')


# In[8]:


seaborn_df = sns.load_dataset('titanic')
seaborn_df.head()


# In[9]:


df = pd.read_csv(r"C:\Users\bhatt\Downloads\hr_data.csv")


# In[10]:


df.head()


# In[12]:


sns.lineplot(x='number_project', y='average_montly_hours', data=df)


# In[13]:


sns.lineplot(x='promotion_last_5years', y='left', data=df)


# In[14]:


plt.figure(figsize=(12,6))
sns.lineplot(x='department', y='left', data=df)


# In[16]:


sns.lineplot(x='number_project', y='average_montly_hours', data=df, hue='left')


# In[23]:


plt.figure(figsize=(12,6))
sns.lineplot(x='number_project', y='average_montly_hours', data=df,
             hue='left',
             style='department',
             legend='full',
            palette='flare')


# # Dist Plot

# plots with bar graph and kernel density estimate

# In[28]:


sns.distplot(df['time_spend_company'])


# In[29]:


sns.distplot(df['left'])


# In[30]:


sns.distplot(df['average_montly_hours'])


# In[31]:


'''
Histograms represent the data distribution by forming bins along the range
of the data and then drawing bars  to show the number of observations that
fall in each bin.
'''


# In[32]:


df.describe()


# In[33]:


bins = [2,3,4,5,6,7,8,9,10]
sns.distplot(df['time_spend_company'], bins=bins)
plt.xticks(bins)


# In[41]:


sns.distplot(df['time_spend_company'], bins = bins,
             rug = True,
             hist_kws={'color':'red', 'edgecolor':'blue', 'linewidth':3, 'alpha': 0.5})
plt.xticks(bins)


# In[46]:


sns.distplot(df['time_spend_company'], bins = bins,
             rug = True,
             hist_kws={'color':'red', 'edgecolor':'blue', 'linewidth':3, 'alpha': 0.5},
             kde_kws={'color':'orange', 'linewidth':3})
plt.xticks(bins)


# In[47]:


sns.distplot(df['time_spend_company'], bins = bins,
             rug = True,
             color='green',
#              hist_kws={'color':'red', 'edgecolor':'blue', 'linewidth':3, 'alpha': 0.5},
#              kde_kws={'color':'orange', 'linewidth':3}
               )
plt.xticks(bins)


# # Scatter Plot

# In[6]:


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')


# In[2]:


titanic_df = sns.load_dataset('titanic')
titanic_df.head()


# In[3]:


sns.scatterplot(x='age', y='fare',
                data = titanic_df)


# In[4]:


sns.scatterplot(x='age', y='fare',
                data=titanic_df, hue='alive')


# In[5]:


plt.figure(figsize=(12,6))
sns.scatterplot(x='age', y='fare', 
                data=titanic_df, hue='alive',
                style='class')
plt.title('Titanic Data Analysis')


# In[6]:


plt.figure(figsize=(12,6))
sns.scatterplot(x='age', y='fare',
                data=titanic_df, hue='alive',
                style='class',
                palette='inferno')
plt.title('Titanic Data Analysis')


# In[7]:


plt.figure(figsize=(12,6))
sns.scatterplot(x='age', y='fare',
                data=titanic_df, hue='alive',
                style='class',
                palette='gist_rainbow', alpha=0.5)
plt.title('Titanic Data Analysis')


# In[9]:


plt.figure(figsize=(12,6))
sns.scatterplot(x='age', y='fare',
                data=titanic_df, hue='alive',
                style='class')
sns.lineplot(x='age', y='fare', data=titanic_df, color='green')
plt.title('Titanic Data Analysis')


# # Bar Plot

# In[12]:


titanic_df.head(1)


# In[11]:


sns.barplot(x='class', y='fare', data=titanic_df)


# A barplot represents as estimate of central tendency for a numeric variable with the height of each rectangle and provides some indication of the uncertainty around that estimate using error bars.

# In[14]:


sns.barplot(x='class', y='fare', data=titanic_df, hue='sex')


# In[16]:


sns.barplot(x='class', y='fare', data=titanic_df, hue='sex', palette='icefire')


# In[17]:


# orient


# In[19]:


sns.barplot(y='class', x='fare', data=titanic_df, 
            hue='sex', palette='inferno', orient='h')


# In[20]:


# estimator


# In[23]:


sns.barplot(x='class', y='fare', data=titanic_df,
            hue='sex', palette='inferno', estimator=np.max)


# In[24]:


# confidence interval


# In[30]:


sns.barplot(x='class', y='fare', data=titanic_df,
            hue='sex', palette='inferno', errorbar=('ci', 100),
            errcolor='#7289da', errwidth=3)


# In[31]:


# saturation


# In[33]:


sns.barplot(x='class', y='fare', data=titanic_df,
            hue='sex', palette='inferno', saturation=0.9)


# # Heatmaps

# In[34]:


flight_df = sns.load_dataset('flights')
flight_df.head()


# In[38]:


flight_df = flight_df.pivot(index='month', columns='year', values='passengers')
flight_df.head()


# In[39]:


plt.figure(figsize=(12,6))
ax = sns.heatmap(flight_df)
ax


# In[40]:


plt.figure(figsize=(14,8))
ax=sns.heatmap(flight_df, annot=True, fmt='d')
ax


# In[43]:


plt.figure(figsize=(14,8))
ax=sns.heatmap(flight_df, annot=True, fmt='d', 
               linecolor='k', linewidths='5')
ax


# In[46]:


plt.figure(figsize=(14,8))
ax=sns.heatmap(flight_df, annot=True, fmt='d',
               linecolor='k', linewidths='5',
               cmap='Blues')


# In[47]:


plt.figure(figsize=(14,8))
sns.heatmap(flight_df, cbar=False)


# In[51]:


grid_kws={"height_ratios":(.4, .05), "hspace": .4}
f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
ax=sns.heatmap(flight_df, cbar_kws={'orientation': 'horizontal'}, ax=ax, 
               cbar_ax=cbar_ax)


# In[53]:


titanic_df.head()


# In[55]:


titanic_df.corr()


# In[3]:


titanic_df = sns.load_dataset('titanic')
plt.figure(figsize=(12,8))
sns.heatmap(titanic_df.corr())


# # Pair Plot

# In[4]:


penguins = sns.load_dataset('penguins')
penguins.head()


# In[7]:


plt.figure(figsize=(12,12))
sns.pairplot(penguins)


# In[8]:


plt.figure(figsize=(12,12))
sns.pairplot(penguins, hue='sex', palette='Reds')


# In[9]:


plt.figure(figsize=(12,12))
sns.pairplot(penguins, hue='species', palette='Blues')


# In[11]:


plt.figure(figsize=(12,12))
sns.pairplot(penguins, kind='hist', palette='Greens')


# In[13]:


plt.figure(figsize=(12,12))
sns.pairplot(penguins, hue='species', diag_kind='hist', palette='rainbow')


# In[14]:


plt.figure(figsize=(12,12))
sns.pairplot(penguins, corner=True)


# In[15]:


plt.figure(figsize=(12,12))
sns.pairplot(penguins, hue='species', markers=['o', 's', 'd'], palette='inferno')


# In[ ]:




