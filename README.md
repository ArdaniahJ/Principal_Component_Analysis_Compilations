# PCA Compilations
Compilations of explorations on PCA upon various datasets

## Colab notebook link
1. [pca_1_hospital_mortality](https://github.com/ArdaniahJ/Principal_Component_Analysis_Compilations/blob/main/PCA1/pca_1_hospital_mortality.py) : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kUwcos7tRANvbmzoun0mjf0FKv-l5rKc?usp=sharing)


<p align="center">
  <img src="https://github.com/ArdaniahJ/Principal_Component_Analysis_Compilations/blob/main/PCA.png" width="700px" height="300px" />
</p>


2. [pca_2_facebook_metrics](https://github.com/ArdaniahJ/Principal_Component_Analysis_Compilations/blob/main/PCA2/pca_2_facebook_metrics.py) : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WD8lqCOQign0a7oNI7-HvyD61BI6nMxj?usp=sharing)


# PCA
PCA is one of the most prominent dimensionality reduction techniques. It is valueable when a dimension needed to be reduced in a dataset while retaining maximum information. 

The __goal__ in PCA is to reduce the number of variables in our dataset. Therefore, the main purpose of PCA is to speed up machine learning. Using fewer variables while still retaining nearly all the original information and variability gives a nice boost to model training, consuming fewer resources and reaching a result faster.

`Eigenvectors and eigenvalues drive PCA at it’s core.` Principal components are uncorrelated with each other where;
+ `Eigenvector` - principal component
+ `Eigenvalues` - variances explained by each eigenvector


## Methods to retain principal components
Unless specified, the `number of principal components will be equal to the number of attributes`.<br> 
These 2 methods below will be used to determine the optimal number of components to retain:
1. `cumulative explained variance`:-
  + is important bcs it allows the model to rank the components (eigenvectors) in order of importance & to focus on the most important one when interpreting the results of the analysis.
  + `pca.explained_variance_ratio_`:- used to get the ration of variance (eigenvalue/total eigenvalues).
  + basically the __percentage__ of the explained variance from `pca.explained_variance__`
2. `scree plot`


## Conditions
There are 3 conditions to adhere when preparing PCA:
1. There must be no null or blank values in the dataset
2. All variables must be in numeric
3. Standardize the variables to have a:
  + mean of 0
  + std var of 1
 
