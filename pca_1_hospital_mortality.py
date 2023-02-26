# -*- coding: utf-8 -*-
"""PCA_1: Hospital_mortality.ipynb

# 3 conditions to adhere when preparing PCA:
1. There must be no null or blank values in the dataset
2. All variables must be in numeric
3. Standardize the variables to have a:
  + mean of 0
  + std var of 1

# Data and Project Goals
This data is from the MIMIC-III database on __in-hospital mortality__. The original purpose of the dataset is to predict whether a patient lives or dies in hospital when in ICU based on the risk factors selected. 

There are a total of __51 variables__ in the data. This project will attempt to reduce these variables down to a more manageable number using PCA. 

These are the goals for this project:
1. Prepare the data for PCA; satisfying the 3 conditions above.
2. Apply the PCA algorithm using sklearn library
3. Determine the optimal number of principal components to retain.

# Data Preprocessing
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# set plot parameters 
sns.set()
custom_params ={"axes.spines.right":False, "axes.spines.top":False}
sns.set_theme(style='ticks', rc=custom_params)
sns.set_palette("Dark2")

# Import data 
pca_raw_df = pd.read_csv('mortality.csv')

pca_raw_df.head()

pca_raw_df.columns

# check the number of variable in the dataset
len(pca_raw_df.columns.values)

"""Now, original variables here is **51**. 

Next, the 3 pre-processing steps:
1. remove any rows that contain N/A
2. select only numberic variables - by calculating the max & assume that a max of 1 is probably not numeric
3. standardize the variables
"""

# 1. remove rows that contain null values
pca_nona_df = pca_raw_df.dropna()

# 2. select only numeric columns
max_all = pca_nona_df.max()
# list() creates a list object - is a collection which is ordered and changeable
max_cols = list(max_all[max_all !=1].index)
# make a new sub df with a max values that != 1
pca_sub_df = pca_nona_df[max_cols]
# make a 2nd sub df from the 1st sub df by dropping the columns that did not fit into the inital logic
pca_sub2_df = pca_sub_df.drop(columns =['group', 'ID', 'gendera'])

# standardise the variables - taken from the (finalized df) 2nd sub df
scaler = StandardScaler()
pca_std = scaler.fit_transform(pca_sub2_df)

# check the variables in finalized dataset (after 3 preprocessing steps)
pca_std.shape

"""Once all pre-processing steps are completed above, 38 variables remain.

# Applying PCA
Now that the data is ready, PCA can be applied and then these 2 methods below will be used to determine the optimal number of components to retain:
1. `cumulative explained variance` 
  + is important bcs it allows the model to rank the components (eigenvectors) in order of importance & to focus on the most important one when interpreting the results of the analysis.
  + `pca.explained_variance_ratio_`- used to get the ration of variance (eigenvalue/total eigenvalues).
  + basically the __percentage__ of the explained variance from `pca.explained_variance__`
2. `scree plot`
"""

# Fit PCA
pca = PCA()
fit_pca = pca.fit_transform(pca_std)

# create a df containing the eigenvalues for each row
pca_fit_df = pd.DataFrame(data=fit_pca)

"""# Plot Cumulative Variance for each component"""

plt.figure(figsize=(15,6))
components = np.arange(1,39, step=1)
variance = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0, 1.1)
plt.plot(components, variance, marker='o', linestyle='--', color='green')

plt.xlabel('Number of Components')
plt.xticks(np.arange(1,39, step=1))

plt.ylabel('Cumulative Explained Variance (%)')
plt.title('The number of commponents needed to explain variance')

# plt.axhline - used to add horizontal line across the axis
plt.axhline(y=0.90, color='r', linestyle='-')
plt.text(0.5, 0.85, '90% variance threshold', color='red', fontsize=16)
plt.text(25, 0.85, "Components needed: "+
         str(np.where(np.cumsum(pca.explained_variance_ratio_)>=0.9)[0][0]), color = "red", fontsize=16)
plt.show()

"""The above plot contains the cumulative explained variance as a **proportion on the y-axis and the number of components on the x-axis**. 

Using a variance threshold of 90%, the above chart helps to determine how many components that should retained from the dataset in order for it to still make sense for any futher modelling.

The explained variance threshold is chosen to be 90% in this project, however, this is not a golden rule as it varies according to each data scientist. 

From the graph, `22 components is needed to retain 90% of the variability (information) in the dataset.` However, that's not much of an improvement over the total number of the variables which is 38.

# Scree Plot
A scree plot is one of the most common ways to determine the number of components to retain.
"""

plt.figure(figsize=(15,6))
components = np.arange(1,39, step=1)
eigenvalues = pca.explained_variance_

plt.plot(components, eigenvalues, marker='o', linestyle='--', color='green')
plt.ylim(0, max(eigenvalues))

plt.ylabel('Eigenvalue')
plt.xlabel('Number of Components')
plt.xticks(np.arange(1,39, step=1))
plt.title('Scree Plot')

# plot the eigenvalue cutoff 
plt.axhline(y=1, color='r', linestyle= '-')
plt.text(0, 0.75, 'Eigenvalue Cutoff', color= 'r', fontsize=14)
plt.text(15, 1.10, 'Components Needed: '+str(np.where(eigenvalues<=1)[0][0]), 
                 color = 'red', fontsize=14)
plt.show()

"""The above scree plot contains; 
+ y-axis: the eigenvalues 
+ x-axis: number of components

As a general rule of thumb, number of components that have an eigenvalue >1 are the one that should be selected. Hence, only 13 components are chosen to be kept. This is very different to the cumulative explain variance plot where the 22 components are retained. 

However, if 13 components are chosen to be retained, this only explains 65% of the variablity. This is much lower than that of the one captured while retaining 22 components (where there are 90% of variability). 

There are methods that can determine the optimal number of components. It is also possible that PCA just isn't a suitable method for this dataset and other dimensionality reduction techniques could be explored. 
"""
