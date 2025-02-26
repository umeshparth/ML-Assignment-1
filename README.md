# ML-Assignment-1

## Q1: Data Preprocessing
You have been provided with a CSV file "Cars93.csv." The given dataset is related to cars and contains 26 columns. In the given dataset, “Price” is the target variable (i.e., the output). The marks distribution according to the tasks are as follows:
1. Assign a type to each of the following features (a) Model, (b) Type, (c) Max. Price and (d) Airbags from the following: ordinal/nominal/ratio/interval scale.
2. Write a function to handle the missing values in the dataset (e.g., any NA, NaN values).
3. Write a function to reduce noise (any error in the feature) in individual attributes.
4. Write a function to encode all the categorical features in the dataset according to the type of variable jointly.
5. Write a function to normalize / scale the features either individually or jointly.
6. Write a function to create a random split of the data into train, validation and test sets in the ratio of [70:20:10].

### Features Classifications
To determine the type of each feature, let's analyze them based on measurement scales:

**(a) Model → Nominal**
It represents the name of the car model, which is just a label without any order.

**(b) Type → Nominal**
Categories like "Sedan," "SUV," "Convertible," etc., have no meaningful ranking.

**(c) Max. Price → Ratio**
Price is a numerical value with a true zero, meaning a car can have no cost. Also, the ratio makes sense (e.g., a $40,000 car is twice as expensive as a $20,000 car).

**(d) Airbags → Ordinal**
This could be categorized as "None," "Driver only," or "Driver & Passenger", which has a meaningful order but no fixed numerical difference.

**_Code_**

```python
def classify_feature(feature_name):
    if df[feature_name].dtype == 'object':  # Categorical features
        if feature_name in ["Model", "Type"]:
            return "Nominal"
        elif feature_name == "Airbags":
            return "Ordinal"
    elif df[feature_name].dtype in ['int64', 'float64']:  # Numerical features
        if feature_name == "Max. Price":
            return "Ratio"
    return "Unknown"
```
### **Final Answer:**
| Feature        | Type   |
|---------------|---------|
| Model         | Nominal |
| Type          | Nominal |
| Max. Price    | Ratio   |
| Airbags       | Ordinal |

### Handling Missing Values
To handle missing values in data set like NA, NaN etc, We calculate the Mean or median and replace missing values with it. 

**Numerical columns:**
Missing values are filled with the mean (default) or median.
**Categorical columns:**
Missing values are filled with the mode (most frequent value).

**_Code_**
```python
def handle_missing_values(df):
    """Fills missing values with median for numerical columns."""
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype in ['float64', 'int64']:
                df[column].fillna(df[column].median(), inplace=True)
            else:
                df[column].fillna(df[column].mode()[0], inplace=True)
    return df
```


![image](https://github.com/user-attachments/assets/bec7ebe4-591f-4de3-8dd5-fe65bdb0c849)

### Reduce Noise
Random or irrelevant data can result in unpredictable situations that are different from what we expected, which is known as noise.
Lets correct the AirBags data as mention in the code.

**_Code_**

```python
def reduce_noise(df):
    """Handles inconsistencies in categorical values by standardizing categories."""
    df['AirBags'] = df['AirBags'].replace({'None': 'No Airbags', 'Driver only': 'Driver', 'Driver & Passenger': 'Both'})
    return df
```

![image](https://github.com/user-attachments/assets/269cdd13-9569-47d6-9b43-03f1ed6c5f18)

### Encode Categorical Features
Here we are going to encode String fields like AirBags also doing encoding based on category like based on type we can have different column with true and false values.

**_Code_**

```python
def encode_categorical(df):
    """Encodes nominal and ordinal categorical features."""
    label_encoders = {}
    # Ordinal Encoding for Airbags
    airbags_order = {'No Airbags': 0, 'Driver': 1, 'Both': 2}
    df['AirBags'] = df['AirBags'].map(airbags_order)
    # One-Hot Encoding for other categorical variables
    df = pd.get_dummies(df, columns=['Type', 'DriveTrain', 'Man.trans.avail', 'Origin'], drop_first=True)
    return df
```
![image](https://github.com/user-attachments/assets/56abf531-44c3-441b-b79d-dd28aa87a485)

### Normalize Features
Here we will scale using Scales numerical features using MinMaxScaler or StandardScaler so that data can be fitted in graph.

**_Code_**

```python
def normalize_features(df, method='minmax'):
    """Scales numerical features using MinMaxScaler or StandardScaler."""
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df
```
![image](https://github.com/user-attachments/assets/d720ee14-0856-4b23-bf46-a48efe843075)

### Split Data
Split data into Train validation and Test sets in the ration of [70:20:10]

**_Code_**

```python
def split_data(df, target_column='Price'):
    """Splits data into train (70%), validation (20%), and test (10%)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
```

## Q2a: Linear Regression Task
Use the “linear_regression_dataset.csv”
Implement the linear regression model to predict the dependency between two variables.
1. Implement linear regression using the inbuilt function “LinearRegression” model in sklearn.
2. Print the coefficient obtained from linear regression and plot a straight line on the scatter plot.
3. Now, implement linear regression without the use of any inbuilt function.
4. Compare the results of 1 and 3 graphically.

### Linear Regression - Inbuild Function

```python
print(f"Sklearn Linear Regression: Coefficient = {model.coef_[0]}, Intercept = {model.intercept_}")
```
Sklearn Linear Regression: Coefficient = 63.1317191283293, Intercept = -42.178608958837785
### Linear Regression - Manual
```python
mean_x = np.mean(X_train_np)
mean_y = np.mean(y_train_np)
m = np.sum((X_train_np - mean_x) * (y_train_np - mean_y)) / np.sum((X_train_np - mean_x)**2)
b = mean_y - m * mean_x
y_pred_manual = m * X_test.to_numpy().flatten() + b
print(f"Manual Linear Regression: Coefficient = {m}, Intercept = {b}")
```
Manual Linear Regression: Coefficient = 63.1317191283293, Intercept = -42.178608958837785

### Compare results graphically

![image](https://github.com/user-attachments/assets/4380c455-e985-4c82-9cb5-24d75d617670)

## Q2b: Logistic Regression
Use the “logistic_regression_dataset.csv”
1. Split the dataset into training set and test set in the ratio of 70:30 or 80:20
2. Train the logistic regression classifier (using inbuilt function: LogisticRegression from sklearn).
3. Print the confusion matrix and accuracy.

```python
# Split data into training and test sets (80:20 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Compute confusion matrix and accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
```
![image](https://github.com/user-attachments/assets/930e7f9f-5a4a-4d8d-b7be-79cee8ffbd1d)

## Q3: SVM
Use the dataset “Bank_Personal_Loan_Modelling.csv”
1. Store the dataset in your google drive and in Colab file load the dataset from your drive.
2. Check the shape and head of the dataset.
3. Age, Experience, Income, CCAvg, Mortgage, Securities are the features and Credit Card is your Target Variable.
i. Take any 3 features from the six features given above
ii. Store features and targets into a separate variable
iii. Look for missing values in the data, if any, and address them accordingly.
iv. Plot a 3D scatter plot using Matplotlib.
4. Split the dataset into 80:20. (3 features and 1 target variable).
5. Train the model using scikit learn SVM API (LinearSVC) by setting the regularization parameter C as C = {0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000}.
i. For each value of C Print the score on test data
ii. Make the prediction on test data
iii. Print confusion matrix and classification report
6. Use gridSearchCV a cross-validation technique to find the best regularization parameters (i.e.: the best value of C).
In the report provide your findings for the output generated for all the kernels used and also describe the changes that happened after changing the regularization hyperparameter.

Dataset Shape: (5000, 14)
Dataset Head:
   ID  Age  Experience  Income  ZIP Code  Family  CCAvg  Education  Mortgage  \
0   1   25           1      49     91107       4    1.6          1         0   
1   2   45          19      34     90089       3    1.5          1         0   
2   3   39          15      11     94720       1    1.0          1         0   
3   4   35           9     100     94112       1    2.7          2         0   
4   5   35           8      45     91330       4    1.0          2         0   

   Personal Loan  Securities Account  CD Account  Online  CreditCard  
0              0                   1           0       0           0  
1              0                   1           0       0           0  
2              0                   0           0       0           0  
3              0                   0           0       0           0  
4              0                   0           0       0           1  

C=0.0001, Accuracy: 0.8950
Classification Report:
              precision    recall  f1-score   support

           0       0.90      1.00      0.94       895
           1       0.00      0.00      0.00       105

    accuracy                           0.90      1000
   macro avg       0.45      0.50      0.47      1000
weighted avg       0.80      0.90      0.85      1000

Confusion Matrix:
[[895   0]
 [105   0]]
--------------------------------------------------
C=0.001, Accuracy: 0.8950
Classification Report:
              precision    recall  f1-score   support

           0       0.90      1.00      0.94       895
           1       0.00      0.00      0.00       105

    accuracy                           0.90      1000
   macro avg       0.45      0.50      0.47      1000
weighted avg       0.80      0.90      0.85      1000

Confusion Matrix:
[[895   0]
 [105   0]]
--------------------------------------------------
C=0.01, Accuracy: 0.8950
Classification Report:
              precision    recall  f1-score   support

           0       0.90      1.00      0.94       895
           1       0.00      0.00      0.00       105

    accuracy                           0.90      1000
   macro avg       0.45      0.50      0.47      1000
weighted avg       0.80      0.90      0.85      1000

Confusion Matrix:
[[895   0]
 [105   0]]
--------------------------------------------------

C=0.1, Accuracy: 0.8950
Classification Report:
              precision    recall  f1-score   support

           0       0.90      1.00      0.94       895
           1       0.00      0.00      0.00       105

    accuracy                           0.90      1000
   macro avg       0.45      0.50      0.47      1000
weighted avg       0.80      0.90      0.85      1000

Confusion Matrix:
[[895   0]
 [105   0]]
--------------------------------------------------
C=1, Accuracy: 0.8950
Classification Report:
              precision    recall  f1-score   support

           0       0.90      1.00      0.94       895
           1       0.00      0.00      0.00       105

    accuracy                           0.90      1000
   macro avg       0.45      0.50      0.47      1000
weighted avg       0.80      0.90      0.85      1000

Confusion Matrix:
[[895   0]
 [105   0]]
--------------------------------------------------

C=10, Accuracy: 0.8950
Classification Report:
              precision    recall  f1-score   support

           0       0.90      1.00      0.94       895
           1       0.00      0.00      0.00       105

    accuracy                           0.90      1000
   macro avg       0.45      0.50      0.47      1000
weighted avg       0.80      0.90      0.85      1000

Confusion Matrix:
[[895   0]
 [105   0]]
--------------------------------------------------

C=100, Accuracy: 0.8950
Classification Report:
              precision    recall  f1-score   support

           0       0.90      1.00      0.94       895
           1       0.00      0.00      0.00       105

    accuracy                           0.90      1000
   macro avg       0.45      0.50      0.47      1000
weighted avg       0.80      0.90      0.85      1000

Confusion Matrix:
[[895   0]
 [105   0]]
--------------------------------------------------

C=1000, Accuracy: 0.8950
Classification Report:
              precision    recall  f1-score   support

           0       0.90      1.00      0.94       895
           1       0.00      0.00      0.00       105

    accuracy                           0.90      1000
   macro avg       0.45      0.50      0.47      1000
weighted avg       0.80      0.90      0.85      1000

Confusion Matrix:
[[895   0]
 [105   0]]
--------------------------------------------------

Best hyperparameter (C): {'C': 0.0001}

## Q4: Decision Tree and Random Forest
Load the IRIS dataset. The dataset consists of 150 samples of iris flowers, each belonging to one of three species (setosa, versicolor, or virginica). Each sample includes four features: sepal length, sepal width, petal length, and petal width.
1. Visualize the distribution of each feature and the class distribution.
2. Encode the categorical target variable (species) into numerical values.
3. Split the dataset into training and testing sets (use an appropriate ratio).
4. Decision Tree Model
i. Build a decision tree classifier using the training set.
ii. Visualize the resulting decision tree.
iii. Make predictions on the testing set and evaluate the model's performance using appropriate metrics (e.g., accuracy, confusion matrix).
5. Random Forest Model
i. Build a random forest classifier using the training set.
ii. Tune the hyperparameters (e.g., number of trees, maximum depth) if necessary.
iii. Make predictions on the testing set and evaluate the model's performance using appropriate metrics and compare it with the decision tree model.

_**Code**_
```Python
# Load the IRIS dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Visualize feature distribution
sns.pairplot(df, hue='species')
plt.show()

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Visualizing the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(dt_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

# Evaluate Decision Tree
y_pred_dt = dt_model.predict(X_test)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
```
![image](https://github.com/user-attachments/assets/cd9494e9-09c5-49db-b139-04d87ce2ebf7)

![image](https://github.com/user-attachments/assets/58e29962-2d66-4de4-8d02-a036d23baa75)

![image](https://github.com/user-attachments/assets/31675c0a-4943-4983-b755-96c22d92beac)

