# Prediction-Models-and-analysis

### INTRODUCTION

In this study, we will examine a dataset including information on individuals with and without diabetes that was released by the National Institute of Diabetes and Digestive and Kidney Diseases in the United States. The objective here is to use the data to decide as to whether or not a certain patient suffers from diabetes. A thorough technique including data cleaning, exploratory data analysis, feature engineering, model selection and assessment, and interpretation will be used to accomplish this aim.

### Dataset Used

The "Pima Indians Diabetes Database" is the sample dataset, and it comprises details on 768 female patients of Pima Indian descent. UCI Machine Learning Repository now houses a dataset amassed by the National Institute of Diabetes and Digestive and Kidney Diseases. The purpose of this data collection is to use parameters like age, body mass index, and blood pressure to forecast the likelihood that a certain patient would acquire diabetes.

- [Download Dataset here](https://data.world/data-society/pima-indians-diabetes-database)

Outcome is a binary variable indicating whether or not the patient was diagnosed with diabetes, and it is used as a goal column. If the number is 1, then the patient did get diabetes; otherwise, the value would be 0.

There are a total of 768 rows in the dataset, with each row representing a different patient. While the dataset as a whole has no missing values, there are a few feature columns with what seem to be impossible or missing values of 0. Since it is very improbable that a patient would have a blood pressure, skin thickness, or insulin level of 0, these values of 0 were probably utilized to represent missing data.

### Data Preprocessing

To prepare raw data for modeling, data preprocessing is an essential part of the machine learning pipeline. In machine learning, model performance is very sensitive to the quality of the training data. To increase the quality of the insights the models produce, it is crucial to subject them to this procedure.
Cleaning, transforming, and normalizing data are all part of the preprocessing phase. 

1. Checked for missing values:
- Attributes including "Glucose," "Bloodpressure," "SkinThickness," "Insulin," and "BMI" in the dataset had invalid 0 values in several columns. Therefore, these 0s were interpreted as missing data. Therefore, the "np.where" function was used to substitute the value "np.nan" for these instances.
- Using the matrix plot function of the "missingno" library, it was discovered that there were gaps in the columns labelled "Insulin," "SkinThickness," and "BMI."
2. Imputting missing values:
- The "KNNImputer" function included in the "sklearn.impute" module was used to fill in the blanks for the missing values in the "Insulin," "SkinThickness," and "BMI" columns. KNNImputer is an algorithm that uses closest neighbours to infer missing data. The default value of 5 neighbours was used.
- Summary statistics were recalculated after missing variables were imputed to ensure accuracy.
3. Checked for duplicate values:
- The "duplicated" function was used to verify the dataset for duplicate values, and it found none..
4. Encoded categorical variables:
- The "LabelEncoder" function in the "sklearn.preprocessing" module was used to convert the "Outcome" column to the digits 0 and 1 for encoding purposes.
5. Splitting dataset:
- Using the "train test split" method in the "sklearn.model selection" package, the dataset was divided into training and testing sets. The original data was split into two sets, with 70% going into the training set and 30% into the testing set.
6. Standardizing data:
- Specifically, the "StandardScaler" method in the "sklearn.preprocessing" module was used to achieve this goal. The data is normalised using the StandardScaler function such that its mean is zero and its standard deviation is one.

![image](https://github.com/Rengoku02/Prediction-Models-and-analysis/assets/103886191/fca2c6c2-3848-4078-a634-38a0c8ec3caf)

- Fig 1:  missing value matrix graphic illustrating how often certain values are absent.


In summary, we encoded categorical variables, separated the dataset into training and testing sets, checked for duplicate values, imputed missing values using KNNImputer, and imputed missing values using KNNImputer. These operations were required to clean the data before modeling.

### Exploratory Data Analysis (EDA)



