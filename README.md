# Travel Order Prediction

### Groups
* < 黃懷萱, 107753002 >

### Goal
Predict whether the order will eventually be made through the information of the travel agency’s order.

### demo
You should provide an example commend to reproduce your result
```
python final.py
```
* any on-line visualization

## Folder organization and its related information

### docs
* Your presentation, 1071_datascience_FP_107753002.pptx, by **Jan. 15**
* Any related document for the final project
  * papers
  * software user guide

### data

* Source
  * Dataset provided by TBrain AI (https://tbrain.trendmicro.com.tw/Competitions/Details/4)
* Input format
  * For more detail information, please see the document in the docs folder.
* Any preprocessing?
  * Replace missing data with "none" because the amount of missing value is small.
  * One hot encoding with category feature.
  * Feature combination

### code

* Which method do you use?
  * RandomForestClassifier
  * GradientBoostingClassifier
  * Lightgbm
  * Xgboost
* What is a null model for comparison?
  * Ramdom guess
* How do your perform evaluation? ie. Cross-validation, or extra separated data
  * Split training data into train data and test data in 8:2 and apply n-fold cross validation.

### results

* Which metric do you use?
  * AUC (Area Under ROC Curve)
* Is your improvement significant?
  * Increase the AUC score from around 0.49(random guess) to around 0.72(xgboost).
* What is the challenge part of your project?
  * Feature selection, Feature combination