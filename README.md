# ML-Assignment-1

## Q1: Data Preprocessing
You have been provided with a CSV file "Cars93.csv." The given dataset is related to cars and contains 26 columns. In the given dataset, “Price” is the target variable (i.e., the output). The marks distribution according to the tasks are as follows:
1. Assign a type to each of the following features (a) Model, (b) Type, (c) Max. Price and (d) Airbags from the following: ordinal/nominal/ratio/interval scale.
2. Write a function to handle the missing values in the dataset (e.g., any NA, NaN values).
3. Write a function to reduce noise (any error in the feature) in individual attributes.
4. Write a function to encode all the categorical features in the dataset according to the type of variable jointly.
5. Write a function to normalize / scale the features either individually or jointly.
6. Write a function to create a random split of the data into train, validation and test sets in the ratio of [70:20:10].

### Handling Missing Values
To handle missing values in data set like NA, NaN etc, We calculate the Mean or median and replace missing values with it. 

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
