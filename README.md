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
		 - feature importances
		 - feature coef
		 - feature weight map
	- Feature selection using SelectFromodel

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
    >* Idea: List all possible values of each parameter and then systematically try all the combinations. 
    
    >* For example, a typical soft-margin SVM classifier equipped with an RBF kernel has at least two hyperparameters that need to be tuned for good performance on unseen data: a regularization constant C and a kernel hyperparameter γ. Both parameters are continuous, so to perform grid search, one selects a finite set of "reasonable" values for each, say C={10,100,1000}, γ = {0.1,0.2,0.5,1.0}. Grid search then trains an SVM with each pair (C, γ) in the Cartesian product of these two sets and evaluates their performance on a held-out validation set (or by internal cross-validation on the training set, in which case multiple SVMs are trained per pair). Finally, the grid search algorithm outputs the settings that achieved the highest score in the validation procedure.
    
    >* Grid search suffers from the curse of dimensionality, but is often embarrassingly parallel because the hyperparameter settings it evaluates are typically independent of each other
    
- Randomized Parameter Optimization
    >* Choose parameter combinations at random
    >* Train & evaluate model with each combination
    >* Choose the combination with the highest accuracy on the validation set
    >* Far more efficient than grid search
    >* No guaranteed to find the best combination
    >* No "intelligence" in the search
- Bayesian Optimization
    >* Exploration of a single combination of hyperparameters is relatively expensive so we want to do as few as possible
    >* We want to learn from each run of the model, so we can then choose the next combination to try as intelligently as possible
    >* We can use a technique called Gaussian Processes to get a better understanding of the underlying relationship between hyperparameter values and model fitness
    > * Bayesian Optimization is often used in applied machine learning to tune the hyperparameters of a given well-performing model on a validation dataset
    >* Bayesian Optimization provides a principled technique based on Bayes Theorem to direct a search of a global optimization problem that is efficient and effective. It works by building a probabilistic model of the objective function, called the surrogate function, that is then searched efficiently with an acquisition function before candidate samples are chosen for evaluation on the real objective function
    >* [YouTube: Bayesian Optimization](https://youtu.be/IAFggDE3tKo)
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
- ExpectationMaximization (EM) Clustering using Gaussian Mixture Models (GMM)
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
	- Examples: Bagging methods, Forests of randomized trees, 

	- By contrast, in boosting methods, base estimators are built sequentially and one tries to reduce the bias of the combined estimator. The motivation is to combine several weak models to produce a powerful ensemble.
	- Examples: AdaBoost, Gradient Tree Boosting, 


## Multiclass and multilabel algorithms
- **Multiclass classification** means a classification task with more than two classes; e.g., classify a set of images of fruits which may be oranges, apples, or pears. Multiclass classification makes the assumption that each sample is assigned to one and only one label: a fruit can be either an apple or a pear but not both at the same time.
- **Multilabel classification** assigns to each sample a set of target labels. This can be thought as predicting properties of a data-point that are not mutually exclusive, such as topics that are relevant for a document. A text might be about any of religion, politics, finance or education at the same time or none of these.
- **Multioutput regression** assigns each sample a set of target values. This can be thought of as predicting several properties for each data-point, such as wind direction and magnitude at a certain location.

## Model persistence
- After training a ML model, it is desirable to have a way to persist the model for future use without having to retrain. 
- It is possible to save a model in scikit-learn by using Pythons built-in persistence model, namely pickle!

## EEG signal processing package in python
- **MNE:** https://mne.tools/stable/auto_examples/index.html
- **PyEEG:** https://github.com/forrestbao/pyeeg
- **EEGLearn:** https://github.com/pbashivan/EEGLearn




