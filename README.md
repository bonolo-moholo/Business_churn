# Business_churn

## Table of contents

- [Project Overview](#project-overview)
- [Tools](#tools)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Split](#data-split)
- [Building a Model](#building-a-model)
- [Recommendations](#recommendation)
- [References](#references)
## Project Overview
The project is about building a machine learning model to predict banking customer churn.The data was collected from kaggle, it had 10000 rows and 14 columns which included: CustomerId, Surname, Creditscore, Geography, Age, Gender, Tenure, Balance, HasCrCard, IsActiveMember,EstimatedSalary and Exited. The model Built is Logistic Regression and this model was chosen because of its speed and interpretability. All features were included in the model except the surname, the row number and the customerID and these were excuded because of their high cardinality.


## Tools

- Python
- Libraries
  1 Pandas (Data manipulation)
  2 Matplotlib (Data Visualizations)
  3 Seaborn (Data Visualizations)
  4 Sklearn (Model Buildind)
  


### Data Cleaning and Preparation

In the initial phase of data cleaning the following tasks were preformed:
- The data was imported/loaded and inspected 
- The data had no missing values 
- The high cardinality column which is row number was dropped

### Exploratory Data Analysis

EDA involved to answer these questions:

- How is the data distributed
   This was checked by checking summary statistics and visualizing th data with a histogram
- What is customer churn by country, by gender, by credit_card avalilabity and active membership
   This was checked by using a bar graph
- How many Customers are there per gender and per country
  This was checked by checking the value counts of each gender and each country and visualized with a bar graph
- What is the relationship between the numerical features and the target
   This was visualized with the use of a heatmap

### Data Split

Categorical features were encoded with one hot encoding
Data was split vertically and horizontally where the Customer churn column named "Exited" was attributed to the target and remaining features attributed to features
and train_test_split from sklearn was used to split the training and testing set with test_set being equal to 20 percent


### Building a model

- A logistic regression model was assigned a variable model and the model was fit to the training data
- The model was used to predict the accuracy of the training and test data
- This is how the model performed:
   Accuracy: 80% for test data
   Precision: 74% on weighted average
   Recall (Sensitivity): 80% on weighted avg
   F1-Score: The recall was 0.74 weighted avg
  

## Recommendations
