# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:11:52 2022

@author: Gopinath
"""

import pandas as pd
import numpy as np
df = pd.read_csv('book.csv',encoding="latin")
df.shape
df.head()
df.sort_values('User.ID')
df1=df.iloc[:,1:]
df1
#number of unique users in the dataset
len(df1)
len(df1['User.ID'].unique())
df1['Book.Rating'].value_counts()
df1['Book.Rating'].hist()
len(df1['Book.Title'].unique())
df1['Book.Title'].value_counts()
user_df = df1.pivot_table(index='User.ID',columns='Book.Title',values='Book.Rating')
user_df
user_df.iloc[0]
user_df.iloc[200]
list(user_df)

#Impute those NaNs with 0 values
user_df.fillna(0, inplace=True)
user_df.shape

from scipy.spatial.distance import cosine
# Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances(user_df.values,metric='cosine')
user_sim.shape
from scipy.spatial.distance import correlation
user_sim = 1 - pairwise_distances( user_df.values,metric='correlation')
user_sim.shape

#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)

#Set the index and column names to user ids
user_sim_df.index   =df1['User.ID'].unique()
user_sim_df.columns = df1['User.ID'].unique()
user_sim
#nullifying diagonal values
np.fill_diagonal(user_sim, 0)
user_sim

#Most Similar Users
user_sim_df.max()
user_sim_df.idxmax(axis=1)
# extract the books which userId 162107 & 276726 have watched
df1[(df1['User.ID']==162107) | (df1['User.ID']==276726)]
# extract the books which userId 276729 & 276726 have watched
df1[(df1['User.ID']==276729) | (df1['User.ID']==276726)]
user_1=df1[df1['User.ID']==276726]
user_2=df1[df1['User.ID']==276729]
pd.merge(user_1,user_2,on='Book.Title',how='outer')

#-------------------------------------------------------------------




