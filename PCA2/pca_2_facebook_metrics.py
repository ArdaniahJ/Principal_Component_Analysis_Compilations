# Data Preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
custom_params ={"axes.spines.right":False, "axes.spines.top":False}
sns.set_theme(style='ticks', rc=custom_params)
sns.set_palette("Dark2")

# Import data 
pca_raw_df = pd.read_csv('dataset_Facebook.csv', sep = ';')
pca_raw_df.head()
pca_raw_df.columns
# check the number of variable in the dataset
len(pca_raw_df.columns.values)

# Three pre-processing steps
pca_raw_df.isnull().sum()

# 1. remove rows that contain null values
pca_nona_df = pca_raw_df.dropna()
# check for pca_nona_df dtypes
pca_nona_df.dtypes

# 2. select only numeric columns
max_all = pca_nona_df.max()
max_cols = list(max_all[max_all !=1].index)
# make a new sub df with a max values that != 1
pca_sub_df = pca_nona_df[max_cols]
# make a 2nd sub df from the 1st sub df by dropping the columns that did not fit into the inital logic
pca_sub2_df = pca_sub_df.drop(columns =['Type'])

pca_sub2_df.describe()
# 3. standardise the variables - taken from the (finalized df)
scaler = RobustScaler()
pca_std = scaler.fit_transform(pca_sub2_df)
# check the variables in finalized dataset (after 3 preprocessing steps)
pca_std.shape

# Applying PCA
# Fit PCA
pca = PCA()
fit_pca = pca.fit_transform(pca_std)
# create a df containing the eigenvalues for each row
pca_fit_df = pd.DataFrame(data=fit_pca)
print(pca.explained_variance_ratio_)
pca_fit_df.head(3)

# Plot Cumulative Variance 
plt.figure(figsize=(15,6))
components = np.arange(1,18, step=1)
variance = np.cumsum(pca.explained_variance_ratio_)
plt.ylim(0.55, 1.0)
plt.plot(components, variance, marker='o', linestyle='--', color='green')
plt.xlabel('Number of Components')
plt.xticks(np.arange(1,18, step=1))
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('The number of commponents needed to explain variance')
plt.axhline(y=0.98, color='r', linestyle='-')
plt.text(0.5, 0.85, '98% variance threshold', color='red', fontsize=16)
plt.text(10, 0.85, "Components needed: "+ str(np.where(np.cumsum(pca.explained_variance_ratio_)>=0.98)[0][0]), color = "red", fontsize=16)
plt.show()

# Scree Plot
plt.figure(figsize=(15,6))
components = np.arange(1,18, step=1)
eigenvalues = pca.explained_variance_
plt.plot(components, eigenvalues, marker='o', linestyle='--', color='green')
plt.ylim(0, max(eigenvalues))
plt.ylabel('Eigenvalue')
plt.xlabel('Number of Components')
plt.xticks(np.arange(1,18, step=1))
plt.title('Scree Plot')
plt.axhline(y=1, color='r', linestyle= '-')
plt.text(0, 0.75, 'Eigenvalue Cutoff', color= 'r', fontsize=14)
plt.text(10, 10, 'Components Needed: '+str(np.where(eigenvalues<=1)[0][0]), color = 'red', fontsize=14)
plt.show()
