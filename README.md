# automate_pricing

## Description of the files contained in this repo
1. ```Data``` contains all the dataset prepared for training and testing.
2. ```query_result_2019-06-26T08_25_26.180Z.csv``` is the csv file containing the data of the sql query result.
3. ```bizongo_po_items.csv``` is the csv file containg modified data.
4. ```sub_items.csv``` is the csv file containing all the bizongo_po_items related to some base products of a particular sub-sub-category. 
5. ```Sagemaker``` includes all the sagemaker notebooks.
6. ```sub_item_train_ipynb``` is the sagemaker code-file taking dates as a string.
7. ```sub_item_train2_ipynb``` is the sagemaker code-file taking date as year entry.
8. ```regression.py``` is the python code-file including all the regression codes.


## Steps to setup the project
1. Clone the repo.
2. Install a python IDE (ex.[spyder](http://www.psych.mcgill.ca/labs/mogillab/anaconda2/lib/python2.7/site-packages/spyder/doc/installation.html))
2. To run the ```regression.py``` file, change the path of sub_item.csv dataset in regression.py file.
3. Run the ```regression.py``` file in a python notebook.

## Amazon Sagemaker setup
1. Create [AWS IAM role](https://console.aws.amazon.com/iam/home?region=ap-south-1#/roles)
2. Create [IAM policy](https://console.aws.amazon.com/iam/home?region=ap-south-1#/policies) 
3. Create a bucket in Amazon [S3](https://s3.console.aws.amazon.com/s3/home?region=ap-south-1)
4. Link the bucket with IAM role and IAM policy.
5. Add the datasets containing the training and test data in Amazon S3 bucket created.
6. Create a [notebook instance](https://ap-south-1.console.aws.amazon.com/sagemaker/home?region=ap-south-1#/notebook-instances) in [Amazon Sagemaker](https://ap-south-1.console.aws.amazon.com/sagemaker/home?region=ap-south-1#/dashboard).
7. Import the jupyter files contained in sagemaker folder of this repo.
8. Tu run the jupyter file, change the dataset path in the file.

## Walkthrough of Steps used in estimation
- **Data Gathering of items**-Gathering of all the specifications related to the item.
- **Grouping of similar items data**-Grouping of items with similar specifications. Grouping is done by taking some base products together. 
- **Preparing Data**-Preparing the dataset to be used as a training dataset.
- **Preparing Data**-Choosing a model
- **Training**-Training the prepared dataset using:
Linear Regression, Multiple Regression, Polynomial Regression, Decision tree Regression, Support Vector Regression, Linear Lerner Regression ( Amazon Sagemaker)
- **Evaluating**-Evaluating the prediction result on test dataset. 

## ML algorithm used
- **Linear Regression**-
[Linear regression](https://en.wikipedia.org/wiki/Linear_regression) is an approach used in machine learning to model a target variable **y** as a linear combination of a vector of explanatory variables **x** and a vector of learned weights **w**.

- **Polynomial Regression**-
[Polynomial regression](https://en.wikipedia.org/wiki/Polynomial_regression)is a form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modelled as an nth degree polynomial in x. Polynomial regression fits a nonlinear relationship between the value of x and the corresponding conditional mean of y, denoted E(y |x), and has been used to describe nonlinear phenomena such as the growth rate of tissues,the distribution of carbon isotopes in lake sediments, and the progression of disease epidemics.

- **Support Vector Regression**-
[Support vector regression](https://en.wikipedia.org/wiki/Support-vector_machine) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting.

- **Decision Tree Regression**-
[Decision Tree regression](https://en.wikipedia.org/wiki/Decision_tree_learning) uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). 

- **Amazon Sagemaker**-
[Amazon SageMaker](https://aws.amazon.com/sagemaker/) machine learning platform provides a number of [built-in algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html) to help customers get up and running training and deploying machine learning models quickly, including [Linear Learner](https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner.html) which can be used for training linear regression models.

