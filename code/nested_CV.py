import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("creditcard.csv")
y = data["Class"]
X = data.drop("Class", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

cv_outer = StratifiedKFold(n_splits=5, random_state=7)

for train_idx, val_idx in tqdm(cv_outer.split(X_train, y_train)):
    train_data, val_data = X_train.iloc[train_idx], X_train.iloc[val_idx]
    train_target, val_target = y_train[train_idx], y_train[val_idx]
    
    model = LogisticRegression(random_state=7)
    cv_inner = StratifiedKFold(n_splits=3, random_state=7)
    params = {'penalty': ['l1', 'l2'], 'class_weight': [None, 'balanced'], 'C': [10**x for x in range(-3,5)]}
    gd_search = GridSearchCV(model, params, scoring='roc_auc', n_jobs=-1, cv=cv_inner).fit(train_data, train_target)
    best_model = gd_search.best_estimator_

    classifier = best_model.fit(train_data, train_target)
    y_pred_prob = classifier.predict_proba(val_data)[:,1]
    auc = metrics.roc_auc_score(val_target, y_pred_prob)

    print("Val Acc:",auc, "Best GS Acc:",gd_search.best_score_, "Best Params:",gd_search.best_params_)
    
  
  # Training final model
  
model = LogisticRegression(random_state=7, C=0.001, class_weight='balanced', penalty='l2').fit(X_train, y_train)
y_pred_prob = model.predict_proba(X_test)[:,1]
print("AUC", metrics.roc_auc_score(y_test, y_pred_prob))
print(metrics.confusion_matrix(y_test, y_pred))
  