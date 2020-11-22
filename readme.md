# Learning feature encoding methods

## Required Packages:

category_encoders

pandas 

scikit-learn


## paper structure 

### Introduction

1. what is (categorical features) feature encoding. 

2. Why we need feature encoding. 

3. relationship between feature encoding and learning model, performance, time/space costs, overfitting. 

4. we want to explorate these effects on one datasets with ** categorical features. 

5. organization of the paper. 

### Feature encoding methods


1. Label Encoding or Ordinal Encoding

2. Target Encoding


3. Binary Encoding

4. Hash Encoding


5. One hot Encoding & Dummy Encoding 

6. Effect Encoding [ref] https://www.researchgate.net/publication/256349393_Categorical_Variables_in_Regression_Analysis_A_Comparison_of_Dummy_and_Effect_Coding


Ordered by space cost. [ref] https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/

### Dataset and DEA

#### Feature types

bin 0-4 : Binary Feature, label encoding
nom 0-9 : Nominal Feature
ord 0-5 : ordinal feature


### Learning Algorithm 

1. logistic regression

2. (Bagging) Random Forest

3. (Boosting) gradient boost (lightbgm,xgboost, gradientboost, etc)

4. MLP

5. KNN

### Metrics 

Metrics for each encoder and model. 

1. test AUC/logloss performance

2. generalization gap (overfitting) train AUC/logloss - test AUC/logloss

3. time costs 

Metrics for each encoder

1. memory costs

### estimated conclusion

1. encoder methods vs model (for different model, do we have any conclusion about encoder methods?)

2. encoder methods vs features (for different features, ... ?)

3. encoder methods vs feature size (for different feature size, ...?)










	


