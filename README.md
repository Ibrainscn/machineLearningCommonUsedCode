Zhenhong Hu
Last Updated: 12/18/2019

-> Machine Learning Fundamentals - - Towards Data Science <-
---
# Pre-processing
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

# Model Evaluation
- Cross-validation
- Confusion Matrix
- Receiver Operating Characteristic (ROC)

# Hyper-parameters Optimization
- Exhaustive Grid Search
- Randomized Parameter Optimization
- Akaike Information Criterion (AIC) 
- Bayes Information Criterion (BIC)

# Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic-Net Regression
- Least-Angle Regression (LARS)
- Logistic Regression
- SVR

# Clustering
- K-Means/K-Medoids
- Mean Shift
- Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
- Expectation–Maximization (EM) Clustering using Gaussian Mixture Models (GMM)
- Agglomerative Hierarchical Clustering
- Ordering Points To Identify Clustering Algorithm (OPTICS)

# Classification
- KNN
- SVM
- Trees
- Random Forests
- Naive-Bayes
- Logistic Regression
- Neural Networks

# Visualizations
- Matplotlib
- Plotly
- Seaborn

# Imputation of missing values
- For various reasons, many real world datasets contain missing values, often encoded as blanks, NaNs or other placeholders. 
- A basic strategy to use incomplete datasets is to discard entire rows and/or columns containing missing values. However, this comes at the price of losing data which may be valuable (even though incomplete). A better strategy is to impute the missing values, i.e., to infer them from the known part of the data.
- Univariate vs. Multivariate Imputation:
	- One type of imputation algorithm is univariate, which imputes values in the i-th feature dimension using only non-missing values in that feature dimension (e.g. impute.SimpleImputer). By contrast, multivariate imputation algorithms use the entire set of available feature dimensions to estimate the missing values (e.g. impute.IterativeImputer).
- Univariate Imputation:
	- The SimpleImputer class provides basic strategies for imputing missing values. Missing values can be imputed with a provided constant value, or using the statistics (mean, median or most frequent) of each column in which the missing values are located. This class also allows for different missing values encodings.
- Multivariate Imputation:
	- A more sophisticated approach is to use the IterativeImputer class, which models each feature with missing values as a function of other features, and uses that estimate for imputation. It does so in an iterated round-robin fashion: at each step, a feature column is designated as output y and the other feature columns are treated as inputs X. A regressor is fit on (X, y) for known y. Then, the regressor is used to predict the missing values of y. This is done for each feature in an iterative fashion, and then is repeated for max_iter imputation rounds. The results of the final imputation round are returned.


# Ensemble methods
- The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability/robustness over a single estimator.
- Two families of ensemble methods are usually distinguished:
	- In averaging methods, the driving principle is to build several estimators independently and then to average their predictions. On average, the combined estimator is usually better than any of the single base estimator because its variance is reduced.
	- Examples: Bagging methods, Forests of randomized trees, …
	- By contrast, in boosting methods, base estimators are built sequentially and one tries to reduce the bias of the combined estimator. The motivation is to combine several weak models to produce a powerful ensemble.
	- Examples: AdaBoost, Gradient Tree Boosting, …

# Model persistence
- After training a ML model, it is desirable to have a way to persist the model for future use without having to retrain. 
- It is possible to save a model in scikit-learn by using Python’s built-in persistence model, namely pickle!






