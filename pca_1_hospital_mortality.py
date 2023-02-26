# Data Preprocessing

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

"""
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

"""
# Applying PCA
1. `cumulative explained variance` 
2. `scree plot`
"""

# Fit PCA
pca = PCA()
fit_pca = pca.fit_transform(pca_std)
# create a df containing the eigenvalues for each row
pca_fit_df = pd.DataFrame(data=fit_pca)

# Plot Cumulative Explained Variance 
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
plt.text(25, 0.85, "Components needed: "+ str(np.where(np.cumsum(pca.explained_variance_ratio_)>=0.9)[0][0]), color = "red", fontsize=16)
plt.show()

# Scree Plot
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
plt.text(15, 1.10, 'Components Needed: '+str(np.where(eigenvalues<=1)[0][0]), color = 'red', fontsize=14)
plt.show()
