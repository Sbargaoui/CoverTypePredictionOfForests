# Packages import
import pandas as pd
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import metrics
import xgboost as xgb

# Loading/training and test data sets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Rearranging the data for further use
train_X = train.drop(['Id','Cover_Type'],axis=1).values
train_y = train.Cover_Type.values
test_X = test.drop('Id',axis=1).values

# Split the training set into training and validation sets
# Here we split into 30% for the test set and 70% for the traning set 
X,X_,y,y_ = cross_validation.train_test_split(train_X,train_y,test_size=0.3)

# Train and predict using the XGBoost algorithm
boost = xgb.XGBClassifier(max_depth=50, n_estimators=500, silent=True, 
	objective="multi:softmax", nthread=-1, gamma=0, min_child_weight=1,
	max_delta_step=0, subsample=1, colsample_bytree=0.3, 
	colsample_bylevel=1, reg_alpha=0, 
	reg_lambda=1, scale_pos_weight=1, seed=0,
	base_score=0.5, missing=None)

# Fitting
boost.fit(X,y)
y_boost = boost.predict(X_)

# Cross validation eval
print metrics.classification_report(y_,y_boost)
print metrics.accuracy_score(y_,y_boost)

# Retraining the model on the whole dataset (100%)
rf.fit(train_X,train_y)
y_test_rf = rf.predict(test_X)

# Finalizing the output
submission = pd.DataFrame({"Cover_Type":y_pred, "Id":id_test}).sort_index(ascending=False,axis=1)
submission.to_csv("submission.csv", index=False)

print('Completed!')