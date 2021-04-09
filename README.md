# Machine Learning Fundamentals - - Towards Data Science
Zhenhong Hu

Last Updated: 12/16/2019

---
## Pre-processing
### Train Test Split
[Cross-validation: evaluating estimator performance in scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)

**Cross-validation iterators for i.i.d. data:**
* **K-Fold CV**: This procedure splits the data into k folds or groups. (k-1) groups will be assigned to train and the remaining group to validate data. This step is repeated for k-steps until all the groups participated in the validation data.

* **Repeated K-Fold**: RepeatedKFold repeats K-Fold n times. It can be used when one requires to run KFold n times, producing different splits in each repetition.

* **Leave One Out (LOO)**: LeaveOneOut (or LOO) is a simple cross-validation. Each learning set is created by taking all the samples except one, the test set being the sample left out. Thus, for  samples, we have  different training sets and  different tests set. This cross-validation procedure does not waste much data as only one sample is removed from the training set．
    - LOO is more computationally expensive than -fold cross validation. In terms of accuracy, LOO often results in high variance as an estimator for the test error. Intuitively, since  of the  samples are used to build each model, models constructed from folds are virtually identical to each other and to the model built from the entire training set.
    
* **ShuffleSplit CV**: Random permutations cross-validation a.k.a. Shuffle & Split: The ShuffleSplit iterator will generate a user defined number of independent train / test dataset splits. Samples are first shuffled and then split into a pair of train and test sets.

**Cross-validation iterators with stratification based on class labels:**

Some classification problems can exhibit a large imbalance in the distribution of the target classes: for instance there could be several times more negative samples than positive samples. In such cases it is recommended to use stratified sampling as implemented in **StratifiedKFold** and **StratifiedShuffleSplit** to ensure that relative class frequencies is approximately preserved in each train and validation fold.
    
* **StatifiedKfold CV**: This procedure is similar to the k-fold CV. Here the dataset is partitioned into k groups or folds such that the validation and train data has an equal number of instances of target class label. This ensures that one particular class is not over present in the validation or train data especially when the dataset is imbalanced.

* **Stratified Shuffle Split**: StratifiedShuffleSplit is a variation of ShuffleSplit, which returns stratified splits, i.e which creates splits by preserving the same percentage for each target class as in the complete set.

**Cross-validation iterators for grouped data:**

The i.i.d. assumption is broken if the underlying generative process yield groups of dependent samples.

Such a grouping of data is domain specific. An example would be when there is medical data collected from multiple patients, with multiple samples taken from each patient. And such data is likely to be dependent on the individual group. In our example, the patient id for each sample will be its group identifier.

In this case we would like to know if a model trained on a particular set of groups generalizes well to the unseen groups. To measure this, we need to ensure that all the samples in the validation fold come from groups that are not represented at all in the paired training fold.

* **Group k-fold**: GroupKFold is a variation of k-fold which ensures that the same group is not represented in both testing and training sets. For example if the data is obtained from different subjects with several samples per-subject and if the model is flexible enough to learn from highly person specific features it could fail to generalize to new subjects. GroupKFold makes it possible to detect this kind of overfitting situations.

* **Leave One Group Out**: LeaveOneGroupOut is a cross-validation scheme which holds out the samples according to a third-party provided array of integer groups. This group information can be used to encode arbitrary domain specific pre-defined cross-validation folds.
    - Each training set is thus constituted by all the samples except the ones related to a specific group. For example, in the cases of multiple experiments, LeaveOneGroupOut can be used to create a cross-validation based on the different experiments: we create a training set using the samples of all the experiments except one.
    
* **Group Shuffle Split**: The GroupShuffleSplit iterator behaves as a combination of ShuffleSplit and LeavePGroupsOut, and generates a sequence of randomized partitions in which a subset of groups are held out for each split.

**Here are the visualization of the above different cross-validation behavior:**
![Visualizing 7 types of cross-validation behavior](https://github.com/Ibrainscn/machineLearningCommonUsedCode/blob/master/image/Visualizing%207%20types%20of%20cross-validation%20behavior.png)
    
**Nested CV** 

Inner loop tune parameters, outer loop train with the optimal parameters. The inner-CV is applied to the (k-1) folds or groups dataset from the outer CV. The set of parameters are optimized using GridSearch and is then used to configure the model. The best model returned from GridSearchCV or RandomSearchCV is then evaluated using the last fold or group. This method is repeated k times, and the final CV score is computed by taking the mean of all k scores.

**Here is a visualization of the nested CV behavior:**
![Nested CV](https://github.com/Ibrainscn/machineLearningCommonUsedCode/blob/master/image/Nested%20CV.jpg)

**The multi-granularity framework for semi-random data partitioning** [paper ref](https://link.springer.com/article/10.1007/s41066-017-0049-2).

This framework involves three levels of granularity as outlined below:
1. Level 1 Data Partitioning is done randomly on the basis of the original data set towards getting a training set and a test set.
2. Level 2 The original data set is divided into a number of subsets, with each subset containing a class of instances. Within each subset (i.e., all instances with a particular class label), data partitioning into training and test sets is done randomly. The training and test sets for the whole data set are obtained by merging all the training and test subsets, respectively.
3. Level 3 Based on the subsets obtained in Level 2, each of them is divided again into a number of subsubsets, where each of the subsubsets contains a subclass (of the corresponding class) of instances. The data partitioning is done randomly within each subsubset. The training and test sets for the whole data set are obtained by merging all the training and test subsubsets, respectively.
            
In this multi-granularity framework, Level 2 is aimed at addressing the class imbalance issue, i.e., to control the distribution of instances by class within the training and test sets. Level 3 is aimed at addressing the issue of sample representativeness, i.e., it is to avoid the case that the training instances are highly dissimilar to the test instances following the data partitioning.
    
![Multi-granularity framework for semi-random data partitioning](https://github.com/Ibrainscn/machineLearningCommonUsedCode/blob/master/image/Multi-granularity%20framework%20for%20semi-random%20data%20partitioning.JPG)


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
	
	- **MICE Package**:
        MICE (Multivariate Imputation via Chained Equations) is one of the commonly used package by R users. Creating multiple imputations as compared to a single imputation (such as mean) takes care of uncertainty in missing values.
        
        MICE assumes that the missing data are Missing at Random (MAR), which means that the probability that a value is missing depends only on observed value and can be predicted using them. It imputes data on a variable by variable basis by specifying an imputation model per variable.
        
        For example: Suppose we have X1, X2….Xk variables. If X1 has missing values, then it will be regressed on other variables X2 to Xk. The missing values in X1 will be then replaced by predictive values obtained. Similarly, if X2 has missing values, then X1, X3 to Xk variables will be used in prediction model as independent variables. Later, missing values will be replaced with predicted values.
        
        By default, linear regression is used to predict continuous missing values. Logistic regression is used for categorical missing values. Once this cycle is complete, multiple data sets are generated. These data sets differ only in imputed missing values. Generally, it’s considered to be a good practice to build models on these data sets separately and combining their results.
        
        Precisely, the methods used by this package are:
        
        * PMM (Predictive Mean Matching)  – For numeric variables
        * logreg(Logistic Regression) – For Binary Variables( with 2 levels)
        * polyreg(Bayesian polytomous regression) – For Factor Variables (>= 2 levels)
        * Proportional odds model (ordered, >= 2 levels)

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

## Time Series 
#### The process:
More information can be found in the post by [Sean Abu](http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/).

![The process of analyze time serires](https://github.com/Ibrainscn/machineLearningCommonUsedCode/blob/master/image/The%20process%20of%20analyz%20time%20serires%20flow%20chart.png)

1. Visualize the data:

    We first want to visualize the data to understand what type of model we should use. Is there an overall trend in your data that you should be aware of? Does the data show any seasonal trends? This is important when deciding which type of model to use. 
    
    If you are using daily data for your time series and there is too much variation in the data to determine the trends, you might want to look at **resampling** your data by month, or looking at the **rolling mean** and **rolling std**.
    
    Another tool to visualize the data is the **seasonal_decompose function in statsmodel**. With this, the **trend and seasonality** become even more obvious.
    
    ```python
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import acf  
    from statsmodels.tsa.stattools import pacf
    decomposition = seasonal_decompose(df.riders, freq=12)  
    trend = decomposition.trend
    seasonal = decomposition.seasonal 
    # The residual values essentially take out the trend and seasonality of the data, making the values independent of time.
    residual = decomposition.residual  
    fig = plt.figure()  
    fig = decomposition.plot()  
    fig.set_size_inches(15, 8)
    ```
2. Stationarize the data:

    What does it mean for data to be stationary?
    
    If a process is stationary, that means it does not change its statistical properties over time, namely its mean and variance. (The constancy of variance is called homoscedasticity) The covariance function does not depend on time; it should only depend on the distance between observations. 
    > So why is stationarity so important?
    >* Because it is easy to make predictions on a stationary series since we can assume that the future statistical properties will not be different from those currently observed. Most of the time-series models, in one way or the other, try to predict those properties (mean or variance, for example). Furture predictions would be wrong if the original series were not stationary. Unfortunately, most of the time series that we see outside of textbooks are non-stationary, but we can (and should) change this.
    >* When running a linear regression the assumption is that all of the observations are all independent of each other. In a time series, however, we know that observations are time dependent. It turns out that a lot of nice results that hold for independent random variables (law of large numbers and central limit theorem to name a couple) hold for stationary random variables. 
    >* So by making the data stationary, we can actually apply regression techniques to this time dependent variable.
    
    There are two ways you can check the stationarity of a time series: 
    
    - The first is by looking at the data. By visualizing the data it should be easy to identify a changing mean or variation in the data. 
    
    - For a more accurate assessment there is the **Dickey-Fuller test**. I won’t go into the specifics of this test, but if the ‘Test Statistic’ is greater than the ‘Critical Value’ than the time series is stationary.
    
    So now we need to transform the data to make it more stationary. There are various transformations you can do to stationarize the data:
    - Logarithmic
    - First Difference
    - Seasonal Difference
    - Seasonal Adjustment
    - You can read more [here](http://people.duke.edu/~rnau/whatuse.htm) about when to use which.


    


### Classical Time Series Forecasting Methods
This cheat sheet demonstrates [11 different classical time series forecasting methods](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/); they are:
1. Autoregression (AR)
2. Moving Average (MA)
3. Autoregressive Moving Average (ARMA)
4. Autoregressive Integrated Moving Average (ARIMA)
5. Seasonal Autoregressive Integrated Moving-Average (SARIMA)
6. Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)
7. Vector Autoregression (VAR)
8. Vector Autoregression Moving-Average (VARMA)
9. Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)
10. Simple Exponential Smoothing (SES)
11. Holt Winter’s Exponential Smoothing (HWES)

### Choosing model order: p and q
Finding appropriate values of p and q in the ARMA(p,q) model can be facilitated by plotting the [partial autocorrelation function (PACF)](https://en.wikipedia.org/wiki/Partial_autocorrelation_function) for an estimate of p, and likewise using the [autocorrelation functions](https://en.wikipedia.org/wiki/Autocorrelation) for an estimate of q. Extended autocorrelation functions (EACF) can be used to simultaneously determine p and q. Further information can be gleaned by considering the same functions for the residuals of a model fitted with an initial selection of p and q.

Brockwell & Davis recommend using [Akaike information criterion (AIC)](https://en.wikipedia.org/wiki/Akaike_information_criterion) for finding p and q. Another possible choice for order determining is the  [Bayesian information criterion (BIC)](https://en.wikipedia.org/wiki/Bayesian_information_criterion) criterion.

[More information on Wikipedia: AR, MA, ARMA, ARIMA ...](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model)

### Python code:
```python
# AR example
from statsmodels.tsa.ar_model import AutoReg
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = AutoReg(data, lags=1)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

# MA example
from statsmodels.tsa.arima.model import ARIMA
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(0, 0, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

# ARMA example
from statsmodels.tsa.arima.model import ARIMA
from random import random
# contrived dataset
data = [random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(2, 0, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

# ARIMA example
from statsmodels.tsa.arima.model import ARIMA
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data), typ='levels')
print(yhat)

# SARIMA example
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

# VAR example
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random
# contrived dataset with dependency
data = list()
for i in range(100):
    v1 = i + random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)
# fit model
model = VAR(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)
```

[An End-to-End Project on Time Series Analysis and Forecasting with Python](https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b)

[Time Series Analysis in Python – A Comprehensive Guide with Examples](https://www.machinelearningplus.com/time-series/time-series-analysis-python/)

[Open Machine Learning Course mlcourse.ai: Topic 9. Part 1. Time series analysis in Python: Basics](https://mlcourse.ai/articles/topic9-part1-time-series/)

[Open Machine Learning Course mlcourse.ai: Topic 9. Part 2. Time series analysis in Python: Predicting the future with Facebook Prophet](https://mlcourse.ai/articles/topic9-part2-prophet/)

[Prophet: Automatic Forecasting Procedure](https://github.com/facebook/prophet): Modeling with Facebook Prophet forecasting package. Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

### How to Convert a Time Series to a Supervised Learning Problem in Python
[Time Series vs Supervised Learning:](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/) Before machine learning can be used, time series forecasting problems must be re-framed as supervised learning problems. From a sequence to pairs of input and output sequences.

### Time series k-fold cross validation
You may be asking how to do cross-validation for time series because time series have this temporal structure and one cannot randomly mix values in a fold while preserving this structure. With randomization, all time dependencies between observations will be lost. This is why we will have to use a more tricky approach in optimizing the model parameters. I don't know if there's an official name to this, but on [CrossValidated](https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection), where one can find all answers but the Answer to the Ultimate Question of Life, the Universe, and Everything, the proposed name for this method is "cross-validation on a rolling basis".

The idea is rather simple -- on a **rolling basis**. We train our model on a small segment of the time series from the beginning until some t , make predictions for the next t+n steps, and calculate an error. Then, we expand our training sample to  t+n  value, make predictions from  t+n  until  t+2∗n , and continue moving our test segment of the time series until we hit the last available observation. As a result, we have as many folds as  n  will fit between the initial training sample and the last observation.

An approach that's sometimes more principled for time series is **forward chaining**. In other words, the "canonical" way to do time-series cross-validation is to **roll through** the dataset, where your procedure would be something like this:
* fold 1 : training [1], test [2]
* fold 2 : training [1 2], test [3]
* fold 3 : training [1 2 3], test [4]
* fold 4 : training [1 2 3 4], test [5]
* fold 5 : training [1 2 3 4 5], test [6]

To make things intuitive, here is an image for same:
![Time series k-fold CV](https://github.com/Ibrainscn/machineLearningCommonUsedCode/blob/master/image/time%20series%20CV.png)


### Time series hv-block cross-validation
There is nothing wrong with using blocks of "future" data for time series cross validation in most situations. By most situations I refer to models for stationary data, which are the models that we typically use. E.g. when you fit an ARIMA(p,d,q), with d>0 to a series, you take d differences of the series and fit a model for stationary data to the residuals.

For cross validation to work as a model selection tool, you need approximate independence between the training and the test data. The problem with time series data is that **adjacent data points are often highly dependent**, so standard cross validation will fail. The remedy for this is to **leave a gap between the test sample and the training samples**, on both sides of the test sample. The reason why you also need to leave out a gap before the test sample is that dependence is symmetric when you move forward or backward in time (think of correlation).

This approach is called hv cross validation (**leave v out, delete h observations on either side of the test sample**) and is described in [this paper](https://doi.org/10.1016/S0304-4076(00)00030-0). For example, this would look like this:
* fold 1 : training [1 2 3 4 5h], test [6]
* fold 2 : training [1 2 3 4h h6], test [5]
* fold 3 : training [1 2 3h h5 6], test [4]
* fold 4 : training [1 2h h4 5 6], test [3]
* fold 5 : training [1h h3 4 5 6], test [2]
* fold 6 : training [h2 3 4 5 6], test [ 1]

Where the h indicates that h observations of the training sample are deleted on that side.



### Granger causality 
Granger causality test is used to determine if one time series will be useful to forecast another.

How does Granger causality test work? It is based on the idea that if X causes Y, then the forecast of Y based on previous values of Y AND the previous values of X should outperform the forecast of Y based on previous values of Y alone.

So, understand that Granger causality should not be used to test if a lag of Y causes Y. Instead, it is generally used on exogenous (not Y lag) variables only.

[More information on Wikipedia: GC](https://en.wikipedia.org/wiki/Granger_causality)


### Hidden Markov Model
Hidden Markov Model (HMM) is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobserved (i.e. hidden) states. HMM are especially known for their application in reinforcement learning and temporal pattern recognition such as speech, handwriting, gesture recognition, part-of-speech tagging, musical score following, partial discharges and bioinformatics.

Terminology in HMM

* States
* Outputs symbols 
* Parameter:
    * Starting probability
    * Transition probability
    * Output/Emission probability
        
The HMM is a generative probabilistic model, in which a sequence of observable X variables is generated by a sequence of internal hidden states Z. The hidden states are not observed directly. The transitions between hidden states are assumed to have the form of a (first-order) Markov chain. They can be specified by the start probability vector π and a transition probability matrix A. The emission probability of an observable can be any distribution with parameters θ conditioned on the current hidden state. The HMM is completely determined by π, A and θ.

There are three fundamental problems for HMMs:

* Given the model parameters and observed data, estimate the optimal sequence of hidden states.
    * This can be solved by the dynamic programming algorithms known as the Viterbi algorithm
* Given the model parameters and observed data, calculate the model likelihood.
    * This can be solved by the dynamic programming algorithms known as the Forward-Backward algorithm
* Given just the observed data, estimate the model parameters.
    * This can be solved by an iterative Expectation-Maximization (EM) algorithm, known as the Baum-Welch algorithm.

* **hmmlearn** implements the Hidden Markov Models: [HMMs python pakage](https://hmmlearn.readthedocs.io/en/latest/tutorial.html)
* [More information on Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)




## Deep Learning
### Basic concept
* Deciding on the `batch size`, number of `epochs`, and `dropout`
    
    `batch size:` The batch size defines the number of samples that will be propagated through the network.
    >For instance, let’s say you have 1000 training samples and you want to set up a batch_size equal to 100. The algorithm takes the first 100 samples (from 1st to 100th) from the training dataset and trains the network. Next, it takes the second 100 samples (from 101st to 200th) and trains the network again. We can keep doing this procedure until we have propagated all samples through the network.
    
    >Advantages of using a batch size < number of all samples:
    >* It requires less memory. Since you train the network using fewer samples, the overall training procedure requires less memory. That’s especially important if you are not able to fit the whole dataset in your machine’s memory.
    >* Typically networks train faster with mini-batches. That’s because we update the weights after each propagation.
    
    >Disadvantages of using a batch size < number of all samples:
    >* The smaller the batch the less accurate the estimate of the gradient will be.
    
    `epochs:` The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.
    > One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters. An epoch is comprised of one or more batches.
    
    >There are no hard and fast rules for selecting batch sizes or the number of epochs, and there is no guarantee that increasing the number of epochs provides a better result than a lesser number.
    
    `dropout:` Dropout is regularization technique to avoid overfitting (increase the validation accuracy) thus increasing the generalizing power.
    > A fully connected layer occupies most of the parameters, and hence, neurons develop co-dependency amongst each other during training which curbs the individual power of each neuron leading to over-fitting of training data.
    >* Generally, use a small dropout value of 20%-50% of neurons with 20% providing a good starting point. A probability too low has minimal effect and a value too high results in under-learning by the network.
    >* Use a larger network. You are likely to get better performance when dropout is used on a larger network, giving the model more of an opportunity to learn independent representations.
    
* Activation Functions

    The activation function is analogous to the build-up of electrical potential in biological neurons which then fire once a certain activation potential is reached. This activation potential is mimicked in artificial neural networks using a probability. Depending upon which activation function is chosen, the properties of the network firing can be quite different.
    >The activation function should do two things:
    >* Ensures not linearity
    >* Ensure gradients remain large through the hidden unit
    
    The general form of an activation function is shown below:
    ![activation function](https://github.com/Ibrainscn/machineLearningCommonUsedCode/blob/master/image/The%20general%20form%20of%20an%20activation%20function.png)
    
    f(.) represents the activation function acting on the weights and biases, producing h, the neural output.
    
    Why do we need non-linearity? 
    >* Technically, we do not need non-linearity, but there are benefits to using non-linear functions.
    If we do not apply an activation function, the output signal would simply be a linear function. A linear function is just a polynomial of one degree. Now, a linear equation is easy to solve but they are limited in their complexity and have less power to learn complex functional mappings from data. A neural network without any activation function would simply be a linear regression model, which is limited in the set of functions it can approximate. 
    >* We want our neural network to not just learn and compute a linear function but something more complicated than that.
    Using a non-linear activation we are able to generate non-linear mappings from inputs to outputs.
    
    Why do we want to ensure we have large gradients through the hidden units?
    >* If we have small gradients and several hidden layers, these gradients will be multiplied during backpropagation. Computers have limitations on the precision to which they can work with numbers, and hence if we multiply many very small numbers, the value of the gradient will quickly vanish. This is commonly known as the vanishing gradient problem and is an important challenge when generating deep neural networks.
    
    Some of the most common choices for activation function are:
    >* Sigmoid
    >* ReLU (rectified linear unit)
    >* Leaky ReLU
    >* Generalized ReLU
    >* MaxOut
    >* Softplus
    >* Tanh
    >* Swish
    
    These activation functions are summarized below:
    ![](https://github.com/Ibrainscn/machineLearningCommonUsedCode/blob/master/image/Summary%20of%20activation%20functions%20for%20neural%20networks.png)
    
    ReLU avoids and rectifies the vanishing gradient problem. Almost all deep learning Models use ReLU nowadays. However, ReLU should only be used within hidden layers of a neural network, and not for the output layer — which should be sigmoid for binary classification, softmax for multiclass classification, and linear for a regression problem.
    [More information about each activation function are here](https://towardsdatascience.com/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98)
    
* Loss Functions

    Loss functions (also called cost functions) are an important aspect of neural networks. 
    
    Maximum Likelihood provides a framework for choosing a loss function in general. As such, the loss function to use depends on the output data distribution and is closely coupled to the output unit.
    
    Cross-entropy and mean squared error are the two main types of loss functions to use when training neural network models.
    However, the maximum likelihood approach was adopted for several reasons, but primarily because of the results it produces. More specifically, neural networks for classification that use a sigmoid or softmax activation function in the output layer learn faster and more robustly using a cross-entropy loss function than using mean squared error.
    
    A summary of the data types, distributions, output layers, and cost functions are given in the table below:  
    ![Table of cost functions](https://github.com/Ibrainscn/machineLearningCommonUsedCode/blob/master/image/cost%20functions.png)







    



## EEG signal processing package in python
- **MNE:** https://mne.tools/stable/auto_examples/index.html
- **PyEEG:** https://github.com/forrestbao/pyeeg
- **EEGLearn:** https://github.com/pbashivan/EEGLearn




