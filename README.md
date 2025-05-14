# Banking Customer churn

## Table of contents

- [Project Overview](#project-overview)
- [Tools](#tools)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Split](#data-split)
- [Building a Model](#building-a-model)
- [Insights](#insights)
- [Recommendations](#recommendation)
- [References](#references)
## Project Overview
The project is about building a machine learning model to predict banking customer churn.The data was collected from kaggle, it had 10000 rows and 14 columns which included: CustomerId, Surname, Creditscore, Geography, Age, Gender, Tenure, Balance, HasCrCard, IsActiveMember,EstimatedSalary and Exited. The model Built initially was Logistic Regression and this model was chosen because of its interpretability and accuracy. All features were included in the model except the surname, the row number and the customerID and these were excuded because of their high cardinality. After evaluating the model it was noted that accuracy was at 71% and overall model performance was affected by class imbalance. Random forest model was then used to improve the accuracy which then increased significantly to 85% but is still open to improvement. 


## Tools

- Python
- Libraries
  - Pandas (Data manipulation)
  - Numpy (Numerical Analysis)
  - Matplotlib (Data Visualizations)
  - Seaborn (Data Visualizations)
  - Sklearn (Model Buildind)
  


### Data Cleaning and Preparation

In the initial phase of data cleaning the following tasks were preformed:

- The data was imported/loaded and inspected 
- The data had no missing values 

### Exploratory Data Analysis

EDA involved to answer these questions:

- How is the data distributed?
   - This was checked by checking summary statistics and visualizing the data with a histogram
- How are the features correlated with the customer churn and with each other?
   - This was done using correlation matrix
- Seeing that age is correlated with churn, what are the age groups that contribute mostly to customer churn?
   - This was done by cutting age groups and visualizing age groups vs churn-"Exited" using  bar chart. Code snipped of age groups:
     ```
     python
     
      #Creating age groups to see how specific age groups influence churn
        bins = [18, 24, 34, 44, 54, 64, 74, 94]  #defining bins
        labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-94'] #defining labes
        df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True) #assigning groups
        df

     ```
- Seeing that balance is also corrlated with churn, How does the availability of funds affect the churn?
    - This was done by creating a HasBalance feature and plotting a bar graph of HasBalance vs Exited to see how churn is affected. Code snipped of Hasbalance
  ```
  python

  #Creating new feature "has balance" to see how no balance accounts affect churn
  df["HasBalance"] = df["Balance"].apply(lambda x: 1 if x > 0 else 0)
  print(df["HasBalance"].value_counts())

  #Analysis of churn rate for HasBalance
  df.groupby("HasBalance")["Exited"].mean()

  ```
    - The distribution of Exited by Balance was created to see whether low or high balance results in churn since churn was observed to be high in accounts with balance
    - A bar graph of balance vs age group was also plotted to see which age groups have high balance
- Since age is correlated with IsActiveMember, Which age groups have high membership?
    - This was plotted using a bar graph for age_group vs IsActiveMember   
- What is customer churn by country, by gender, by credit_card avalilabity and active membership
   - This was checked by using a bar graph
- How many Customers are there per gender and per country
  - This was checked by checking the value counts of each gender and each country and visualized with a bar graph


### Data Split

- Data was split to the target and features where the Customer churn column named "Exited" was attributed to the target and remaining features attributed to features
- Features were then split to numerical, categorical and binary
- The numerical, categorical and binary features were transformed by scaling numerical features to ensure numerical stability and categorical featurers were hot encoded to binary features, binary features were left to remain as they are in the following way"

  ```

  python

  #Splitting Features and target
  X = df.drop(columns=['CustomerId', 'Surname', 'Exited', 'RowNumber', 'Age_Group', 'HasBalance'])  # Remove irrelevant columns
  y = df['Exited']

  #Feature types
  num_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Tenure', 'NumOfProducts']
  cat_features = ['Geography','Gender']
  binary_features = ['HasCrCard', 'NumOfProducts', 'IsActiveMember']

  # Create column transformer (scaling + encoding)
  preprocessor = ColumnTransformer([
    ('num_scaler', StandardScaler(), num_features),  # Standardize numerical data
    ('cat_encoder', OneHotEncoder(drop='first'), cat_features)  # One-hot encode categorical data
  ], remainder='passthrough') #Binary data should remain as it is

  ```
- The data was then split to train set and test set by train_test_split from sklearn where train set is 80% and test set is 20%
  
### Building a model

- A pipeline that included the transformers(StandardScaler, OneHotEncoder, passthrough, SMOTE  and the logistic regression) was created
- The pipeline was then fitted to the data
- Predictions were then made on the train and test data
- The model was evaluated using the Accuracy_score and the classification report
- This is how the model performed:
   - Accuracy: 71.7% of the total predictions were correct
   - Precision: 90% of all Non Exited Customers were correct and of all predicted Exited customers only 39% actually exited
   - Recall (Sensitivity): Of all actual Not Exited customers 72% were correctly predicted and of all actual Exited 70% were identified correctly
   - F1-Score: 0.50 there is moderate balance between precision and recall
- The confusion matrix: True Negatives = 1151, False Positives = 442, False Negatives = 122, True Positives = 285
- ROC curve were plotted to validate further and it was at 78%
- The Feature importance was plotted to check features that contribute the most to churn and this will be described under insights.
- Seeing that performance still needed improvement, a different algorithm was explored which is Random Forest and the model improved as followes:
   - Accuracy: improved to 85.95% 
   - Precision: 87% of all Non Exited Customers were correct and of all predicted Exited customers  recall improved to 77%
   - Recall (Sensitivity): Of all actual Not Exited customers 97% were correctly predicted and of all actual Exited 44% were identified correctly
   - F1-Score: is now 0.92 for Non-Exited and 0.56 for Exited
- The confusion matrix: True Negatives = 1540, False Positives = 53, False Negatives = 228, True Positives = 179
- ROC-AUC was 85%

  ### Insights

- The highest contributing feature to churn is low active membership
- Followed by Geographical location in Germany, showing more customers churning most at Germany
- Followed by older age groups, the age_group vc churn graph shows this is from 45-75 with the highest peak at 55-64
- The female gender churn follow after age group
- The balance feature contributes far lesser than the 4 features above but it also contributes significantly to churn, with customers with higher balances churning the most according to the box plot created in exploratory data analysis.
     
### Recommendations

- Customer engagement needs to improve in order to improve low active membership by:
  - Introducing loyalty programmes to increase engagement
  - By running promotions to inactive members
  - By offering content that promote financial literacy
- Retention strategies are required to keep German Customers
  - First an indepth investigation needs to be conducted to find out why customers churn at such a high rate
  - When the factors contributing to this have been found, they need to be addressed swiftly and ensure an ongoing strategy is in place to deal with them
  - Competition in that area needs to assessed and strategies should be in place to stay ahead of the competition
  - Customer complaints needs to be collected to check customer feedback on products offered and what needs to be improved
- Churn Among older age groups should be looked in to
  - Products tailored for older age groups should be provided or improved such as retirement services
  - A survey of service satisfaction should be conducted as these age groups also have high balances and the problems from feedback should be addressed
  - Accessibility features should be improved, online services should be made easy to use for these age groups
  - Consultants must always be made available to address any issues faced by these age groups as they mostly prefer talking to real people other than bots
- Churn among female genders need to be improved
  - Investigate whether there are products or policies affecting the female customers negatively and improve on this
  - Ensure the banking services cater for specific financial needs for female customers
- Investigate why high balance customers churn
  - Investigate whether high balance customers cold be recieving better offers or benefits from competition and how premium service can be improved
  - Provide exclusive perks for high balance customers
  - Provide better interest rates for investments and savings  to keep high balance customers

### References

Kaggle,
ChatGPT
