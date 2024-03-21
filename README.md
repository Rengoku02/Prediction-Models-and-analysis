# Prediction-Models-and-analysis

## Table of content

- [Introduction](#introduction)
- [Dataset Used](#dataset-used)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Prediction Models](#prediction-models)
- [Conclusion](#conclusion)


## Introduction

In this study, we will examine a dataset including information on individuals with and without diabetes that was released by the National Institute of Diabetes and Digestive and Kidney Diseases in the United States. The objective here is to use the data to decide as to whether or not a certain patient suffers from diabetes. A thorough technique including data cleaning, exploratory data analysis, feature engineering, model selection and assessment, and interpretation will be used to accomplish this aim.

## Dataset Used

The "Pima Indians Diabetes Database" is the sample dataset, and it comprises details on 768 female patients of Pima Indian descent. UCI Machine Learning Repository now houses a dataset amassed by the National Institute of Diabetes and Digestive and Kidney Diseases. The purpose of this data collection is to use parameters like age, body mass index, and blood pressure to forecast the likelihood that a certain patient would acquire diabetes.

- [Download Dataset here](https://data.world/data-society/pima-indians-diabetes-database)

Outcome is a binary variable indicating whether or not the patient was diagnosed with diabetes, and it is used as a goal column. If the number is 1, then the patient did get diabetes; otherwise, the value would be 0.

There are a total of 768 rows in the dataset, with each row representing a different patient. While the dataset as a whole has no missing values, there are a few feature columns with what seem to be impossible or missing values of 0. Since it is very improbable that a patient would have a blood pressure, skin thickness, or insulin level of 0, these values of 0 were probably utilized to represent missing data.

## Data Preprocessing

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

## Exploratory Data Analysis

The goal of exploratory data analysis (EDA) is to uncover hidden patterns and relationships in large data sets. The purpose of EDA is to learn about the data and the connections between different factors. Exploratory data analysis (EDA) makes use of statistical and visual methods to sift through data and draw conclusions. Finding outliers, missing values, and other data abnormalities that might compromise the efficacy of statistical models or machine learning algorithms is a crucial part of the data analysis process.

The first step in EDA is often a visual examination of the data in the form of graphs and charts. Common methods of data visualisation include histograms, scatter plots, box plots, and heat maps. A variable's distribution may be visualized using various graphs and charts, such as histograms, scatter plots, box plots, and heat maps, to name a few.
Next, summary statistics like mean, median, mode, standard deviation, and variance are computed based on the data that has been shown. Using these metrics, you can get a feel for the data as a whole and see any outliers or unexpected readings more easily.
Finding patterns and correlations between variables is the next stage in EDA. To investigate the connection between two variables, correlation analysis is often used. How closely two variables are linked is quantified using a correlation study. A perfect positive connection is represented by a correlation value of 1, while a perfect negative relationship is represented by a correlation coefficient of -1. There is no connection between the variables if the correlation coefficient is 0.

Regression analysis is another vital tool in EDA. A dependent variable's connection with one or more independent variables may be investigated through regression analysis. One typical method for doing this is linear regression. A straight line is used to fit the data, and the strength of the link between the variables is calculated.
Finding and dealing with outliers and missing data is another part of EDA. A data set may have "missing values" if a certain value is not there. You may deal with them in one of two ways: either throw out the rows that include them or fill in the blanks with an estimate. Values that are very out of line with the rest of the data set are known as outliers. Statistical methods may be used to locate them, and then they can be eliminated or transformed.

```python
colors = ['#76EEC6','#E3CF57','#458B74','#00FFFF']
fig = go.Figure(data=go.Splom(dimensions=[dict(label=col,
                                               values=df[col]) for col in 
                                          df[num_cols].select_dtypes(include = ['int', 'float']).columns
                                         ],
                showupperhalf = True, 
                text = df['Outcome'],
                marker = dict(color = [colors[i] for i in df['Outcome']. \
                                     astype('category').cat.codes],
                            showscale = False,
                            opacity = 0.65)
                             )
               )
fig.update_layout(title = {'text': 'Pairwise Relationships by Outcome',
                          'xanchor': 'center',
                          'yanchor': 'top',
                          'x': 0.5,
                          'y': 0.95},
                  width = 950,
                  height = 950,
                  template = 'plotly_dark')
iplot(fig)
```
- Fig 2 : The code above generates a scatter plot matrix (SPLOM) for the diabetes dataset to display the correlations between the numerical variables in terms of the outcome variable.

![image](https://github.com/Rengoku02/Prediction-Models-and-analysis/assets/103886191/5dc61528-2394-4eed-a879-9385c255872b)

## Prediction Models

Algorithms used for data analysis and prediction are known as prediction models. Numerous industries, including business, medicine, marketing, and others, make extensive use of these models. A prediction model's primary function is to reliably foretell future outcomes using existing data.
```python
random_state = 42
models = [GaussianNB(), 
          DecisionTreeClassifier(random_state = random_state),
          SVC(random_state = random_state, probability = True),
          RandomForestClassifier(random_state = random_state),
          LogisticRegression(random_state = random_state),
          GradientBoostingClassifier(random_state = random_state),
for model in models:
model_performance(model)
```
- The aforementioned code is used to compare the accuracy of 10 different diabetes categorization models. Cross-validation is used to assess the models' efficacy once they have been instantiated using the default hyperparameters.

### 1.Gaussian Naive Bayes:

- The accuracy of this model is often about 75%.
- The model's ability to accurately detect positive cases, as measured by the accuracy score, is close to 61%.
- About 79% of really positive examples are properly recognized by the model, as measured by the recall score.
- The F1-score, a harmonic mean of the accuracy and recall scores, is about 68%.
- The area under the receiver operating characteristic curve, which evaluates how well the model separates positive and negative examples, is close to 76 percent.

![image](https://github.com/Rengoku02/Prediction-Models-and-analysis/assets/103886191/ff02f1ce-369b-4274-9baf-0b000d96e084)

- Fig 3 : output of Gaussian Naive Bayes

### 2.Decision Tree: 

- With this model, we may expect an average accuracy of around 71%.
- Precision: About 64% is the overall precision rating.
- About 60% of the information has been recalled.
- The F1-score is around 62%.
- The area under the curve (ROC) for the AUC is close to 69 percent.

![image](https://github.com/Rengoku02/Prediction-Models-and-analysis/assets/103886191/c3ec8ba0-29aa-424b-9ded-8136c93eb268)

- Fig 4 : output of  Decision Tree

### 3.Support Vector Classifier:

- The SVM model has an average accuracy of around 77%.
- Precision: The precision rating is around 66%.
- The recall percentage is around 75%.
- In terms of the F1-score, almost 70% is achieved.
- The area under the curve (ROC) for this measure is around 80%.

![image](https://github.com/Rengoku02/Prediction-Models-and-analysis/assets/103886191/93cabc97-f09b-445c-ad90-2311e4a137f9)

- Fig 5 : SVM Output

### 4.Random Forest:

- In terms of accuracy, the random forest model gets close to 76% on average.
- Precision: The precision rating is around 67%.
- About 71% of information has been recalled.
- As for the F1-score, it hovers around 69%.
- In terms of ROC AUC, the score is close to 81%.

![image](https://github.com/Rengoku02/Prediction-Models-and-analysis/assets/103886191/4a565644-09c1-4c1f-a096-729d55ab67bf)

- Fig 6 : Random Forest output

### 5.Logistic Regression:

- The overall accuracy of the logistic regression model is close to 76%.
- Precision: The precision rating is around 66%.
- About 73% of the information has been recalled.
- As for the F1-score, it comes in at about 69%.
- The area under the curve (ROC) for the AUC is about 82%.

![image](https://github.com/Rengoku02/Prediction-Models-and-analysis/assets/103886191/67120248-ad91-4f49-b73a-5d1c4557cf1b)

Fig 7 : output of  Logistic Regression

In terms of accuracy, precision, recall, F1-score, and ROC AUC, the findings show that ensemble-based models such as Random Forest and Gradient Boosting perform better than the other models. These models are superior in their ability to forecast outcomes because they can identify subtleties in the data. The most precise results are obtained by XGBoost among the ensemble models, followed by LightGBM and CatBoost. Overall, the performance of these models is rather high, hence it is encouraged that they be investigated and optimised further.

It's worth noting that the models' efficacy may change based on the data and the task at hand. Therefore, it is recommended to try out several models, hyperparameter tweaking, and extra feature engineering methods to further improve the performance of the selected model.

## Conclusion

To sum up, the process for diabetes prediction presented in this research is thorough and well-organized. Preprocessing data, doing exploratory data analysis, developing features, training and evaluating models, and interpreting results are all included. When numerous machine learning algorithms are used, consumers are able to compare and contrast all of the available options and choose the one that works best for them. This article is useful for both novice and experienced practitioners in the fields of machine learning and healthcare analytics since it provides easy-to-understand explanations, code samples, and visualisations.









