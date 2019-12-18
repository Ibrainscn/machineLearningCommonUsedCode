# Machine Learning Fundamentals - - Towards Data Science
Zhenhong Hu

Last Updated: 12/16/2019

---
## Pre-processing
- Feature Extraction
- Scaling and Normalization
- Feature Selection
	- Removing features with low variance
	- Univariate feature selection
	- Recursive feature elimination
		 - feature_importances_
		 - feature_coef_
		 - feature_weight_map_
	- Feature selection using SelectFromModel

- Dimensionality Reduction
	- Principal Component Analysis (PCA)
	- Linear Discriminat Aanalysis (LDA)

## Model Evaluation
- Cross-validation
- Confusion Matrix
- Receiver Operating Characteristic (ROC)
- Precision-Recall 

## Hyper-parameters Optimization
- Exhaustive Grid Search
- Randomized Parameter Optimization
- Akaike Information Criterion (AIC) 
- Bayes Information Criterion (BIC)

## Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic-Net Regression
- Least-Angle Regression (LARS)
- Logistic Regression
- SVR

## Clustering
- K-Means/K-Medoids
- Mean Shift
- Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
- Expectation–Maximization (EM) Clustering using Gaussian Mixture Models (GMM)
- Agglomerative Hierarchical Clustering
- Ordering Points To Identify Clustering Algorithm (OPTICS)

## Classification
- KNN
- SVM
- Trees
- Random Forests
- Naive-Bayes
- Logistic Regression
- Neural Networks

## Visualizations
- Matplotlib
- Plotly
- Seaborn

## Imputation of missing values
- For various reasons, many real world datasets contain missing values, often encoded as blanks, NaNs or other placeholders. 
- A basic strategy to use incomplete datasets is to discard entire rows and/or columns containing missing values. However, this comes at the price of losing data which may be valuable (even though incomplete). A better strategy is to impute the missing values, i.e., to infer them from the known part of the data.
- Univariate vs. Multivariate Imputation:
	- One type of imputation algorithm is univariate, which imputes values in the i-th feature dimension using only non-missing values in that feature dimension (e.g. impute.SimpleImputer). By contrast, multivariate imputation algorithms use the entire set of available feature dimensions to estimate the missing values (e.g. impute.IterativeImputer).
- Univariate Imputation:
	- The SimpleImputer class provides basic strategies for imputing missing values. Missing values can be imputed with a provided constant value, or using the statistics (mean, median or most frequent) of each column in which the missing values are located. This class also allows for different missing values encodings.
- Multivariate Imputation:
	- A more sophisticated approach is to use the IterativeImputer class, which models each feature with missing values as a function of other features, and uses that estimate for imputation. It does so in an iterated round-robin fashion: at each step, a feature column is designated as output y and the other feature columns are treated as inputs X. A regressor is fit on (X, y) for known y. Then, the regressor is used to predict the missing values of y. This is done for each feature in an iterative fashion, and then is repeated for max_iter imputation rounds. The results of the final imputation round are returned.

## Novelty and Outlier Detection
- Outlier detection and novelty detection are both used for anomaly detection, where one is interested in detecting abnormal or unusual observations. Outlier detection is then also known as unsupervised anomaly detection and novelty detection as semi-supervised anomaly detection. In the context of outlier detection, the outliers/anomalies cannot form a dense cluster as available estimators assume that the outliers/anomalies are located in low density regions. On the contrary, in the context of novelty detection, novelties/anomalies can form a dense cluster as long as they are in a low density region of the training data, considered as normal in this context.
- Outlier Detection Methods
  - Robust Covariance
  - One-Class SVM
  - Isolation Forest
  - Local Outlier Factor
- Novelty Detection Methods
  - Local Outlier Factor 

## Ensemble methods
- The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability/robustness over a single estimator.
- Two families of ensemble methods are usually distinguished:
	- In averaging methods, the driving principle is to build several estimators independently and then to average their predictions. On average, the combined estimator is usually better than any of the single base estimator because its variance is reduced.
	- Examples: Bagging methods, Forests of randomized trees, …
	- By contrast, in boosting methods, base estimators are built sequentially and one tries to reduce the bias of the combined estimator. The motivation is to combine several weak models to produce a powerful ensemble.
	- Examples: AdaBoost, Gradient Tree Boosting, …

## Multiclass and multilabel algorithms
- **Multiclass classification** means a classification task with more than two classes; e.g., classify a set of images of fruits which may be oranges, apples, or pears. Multiclass classification makes the assumption that each sample is assigned to one and only one label: a fruit can be either an apple or a pear but not both at the same time.
- **Multilabel classification** assigns to each sample a set of target labels. This can be thought as predicting properties of a data-point that are not mutually exclusive, such as topics that are relevant for a document. A text might be about any of religion, politics, finance or education at the same time or none of these.
- **Multioutput regression** assigns each sample a set of target values. This can be thought of as predicting several properties for each data-point, such as wind direction and magnitude at a certain location.

## Model persistence
- After training a ML model, it is desirable to have a way to persist the model for future use without having to retrain. 
- It is possible to save a model in scikit-learn by using Python’s built-in persistence model, namely pickle!

## EEG signal processing package in python
- **MNE:** https://mne.tools/stable/auto_examples/index.html
- **PyEEG:** https://github.com/forrestbao/pyeeg
- **EEGLearn:** https://github.com/pbashivan/EEGLearn




