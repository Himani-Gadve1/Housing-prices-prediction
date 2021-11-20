# Housing-prices-prediction

I have used housing data set found on on Kaggle. This is a simple data set, but it serves the purpose of this project. The details of this data set can be found [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)

I have removed some of the columns that contains too many missing values. I have also removed the remaining missing values for you, to save you some work. The goal again is to predict the housing price using the predictors.

## Part I: Dimension Reduction

For dimension reduction, there is a technique called Multiple Correspondence Analysis (MCA) that can be used for categorical data. MCA is an extension of Correspondance Analysis (CA). Your tasks for dimension reduction are as follow:

1. Apply MCA onto the data set. Here are some guidelines.
•	Apply PCA on the numerical features, and MCA on the categorical features.

•	One thing about MCA, and probably with many other methods that deal with
categorical data, is if there’s a new class showing up in the test data, the algorithm
doesn’t know how to handle it (you will get an error message that says dimension
mismatch). Therefore before you apply any MCA fitting and transform, make
sure that each categorical variables in your training set and testing set have the same set of levels. Remove any columns that don’t match. I have written a short
script for you to perform this task.

•	Perform regression using the reduced features, and report your evaluation result (i.e. testing set).

•	Be sure to tune your PCA and MCA. It may be easiest to tune using a for loop
plus a basic train / valid / test split, rather than a CV (though you are very
welcomed to try coding it).

•	Follow the standard procedure on tuning hyperparameters.

2. As a baseline comparison, also perform regression on the original data set. You may
use Ridge regression, but not LASSO.
•	Since you have already splitted the data into training and testing, you can just use those.

•	You can create dummy variables based on the categorical features you separated
from step 1. Given that both the training and testing (categorical sets) should
have the same columns and levels, there shouldn’t be a problem in creating the
dummy variables separately.

3. To help you get started, the Python library prince can perform MCA. A simple tutorial of this library can be found [here](https://github.com/MaxHalford/prince#multiple-correspondence-analysis-mca)

4. Describe your findings.

To give you some hint, here is what you can do with respect to tuning the hyper-	parameters:
	1. Split the data into 3 pieces (training, validation, testing).
	2. Apply the provided code on the training and validation sets so the categorical features
	are the same.
	3. Tune the model.
	4. Now reapply the code, but on the training + validation sets combined, and the test
	set.
	5. Fit your final model, perform prediction.


## Part II: Clustering Analysis

While we perform dimension reduction separately for numerical and categorical data, there are methods that can perform clustering analysis with numerical and categorical data combined. As usual, the most important aspect is the distance metric to use. For mixed data
types, researchers have proposed to use the Gower distance. The Gower distance is essentially a special distance metric that measures numerical data and categorical data separately, then combine them to form a distance calculation. Here are your tasks for clustering:

1. Tutorials on Gower distance can be found here : https://rstudio-pubs-static.s3.amazonaws.com/423873_adfdb38bce8d47579f6dc916dd67ae75.html
https://medium.com/analytics-vidhya/gowers-distance-899f9c4bd553/

2. Compute the Gower distance of the full predictors set, i.e. no train/test split. There
is a Python library called gower you can use [link here](https://pypi.org/project/gower/)
3. Apply K-medoids using the gower distance matrix as input. The k-medoid function
allows the user to input a distance matrix instead of the raw data. Please read the
documentation for more details. You may use any number of clusters.
4. As a simple exercise, you will compare the clustering result with the ’ground truth’.

	(a) Your clustering result will only tell you which observations belong to cluster k.
So first create an array that records the cluster membership of each observation.
	(b) Bin the response variable (of the original data set) into the number of categories
you used for k-medoids (for example, if you choose 5 clusters, bin the response
into 5 groups). Use the pandas.qcut function so each group has roughly the same
number of observations.
	(c) Compute the normalized mutual information (NMI) between your cluster-
ing results and the binned categories. (NMI is a metric to measure clustering
performance.
