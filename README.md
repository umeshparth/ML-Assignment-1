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
Lets correct the AirBags data as mentione in the code.

**_Code_**

```python
def reduce_noise(df):
    """Handles inconsistencies in categorical values by standardizing categories."""
    df['AirBags'] = df['AirBags'].replace({'None': 'No Airbags', 'Driver only': 'Driver', 'Driver & Passenger': 'Both'})
    return df
```

![image](https://github.com/user-attachments/assets/bec7ebe4-591f-4de3-8dd5-fe65bdb0c849)
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Manufacturer</th>
      <th>Model</th>
      <th>Type</th>
      <th>Min.Price</th>
      <th>Price</th>
      <th>Max.Price</th>
      <th>MPG.city</th>
      <th>MPG.highway</th>
      <th>AirBags</th>
      <th>DriveTrain</th>
      <th>...</th>
      <th>Fuel.tank.capacity</th>
      <th>Passengers</th>
      <th>Length</th>
      <th>Wheelbase</th>
      <th>Width</th>
      <th>Turn.circle</th>
      <th>Rear.seat.room</th>
      <th>Luggage.room</th>
      <th>Weight</th>
      <th>Origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Acura</td>
      <td>Integra</td>
      <td>Small</td>
      <td>12.9</td>
      <td>15.9</td>
      <td>18.8</td>
      <td>25</td>
      <td>31</td>
      <td>Driver</td>
      <td>Front</td>
      <td>...</td>
      <td>13.2</td>
      <td>5</td>
      <td>177</td>
      <td>102</td>
      <td>68</td>
      <td>37</td>
      <td>26.5</td>
      <td>11.0</td>
      <td>2705</td>
      <td>non-USA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Acura</td>
      <td>Legend</td>
      <td>Midsize</td>
      <td>29.2</td>
      <td>33.9</td>
      <td>38.7</td>
      <td>18</td>
      <td>25</td>
      <td>Both</td>
      <td>Front</td>
      <td>...</td>
      <td>18.0</td>
      <td>5</td>
      <td>195</td>
      <td>115</td>
      <td>71</td>
      <td>38</td>
      <td>30.0</td>
      <td>15.0</td>
      <td>3560</td>
      <td>non-USA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Audi</td>
      <td>90</td>
      <td>Compact</td>
      <td>25.9</td>
      <td>29.1</td>
      <td>32.3</td>
      <td>20</td>
      <td>26</td>
      <td>Driver</td>
      <td>Front</td>
      <td>...</td>
      <td>16.9</td>
      <td>5</td>
      <td>180</td>
      <td>102</td>
      <td>67</td>
      <td>37</td>
      <td>28.0</td>
      <td>14.0</td>
      <td>3375</td>
      <td>non-USA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Audi</td>
      <td>100</td>
      <td>Midsize</td>
      <td>30.8</td>
      <td>37.7</td>
      <td>44.6</td>
      <td>19</td>
      <td>26</td>
      <td>Both</td>
      <td>Front</td>
      <td>...</td>
      <td>21.1</td>
      <td>6</td>
      <td>193</td>
      <td>106</td>
      <td>70</td>
      <td>37</td>
      <td>31.0</td>
      <td>17.0</td>
      <td>3405</td>
      <td>non-USA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>535i</td>
      <td>Midsize</td>
      <td>23.7</td>
      <td>30.0</td>
      <td>36.2</td>
      <td>22</td>
      <td>30</td>
      <td>Driver</td>
      <td>Rear</td>
      <td>...</td>
      <td>21.1</td>
      <td>4</td>
      <td>186</td>
      <td>109</td>
      <td>69</td>
      <td>39</td>
      <td>27.0</td>
      <td>13.0</td>
      <td>3640</td>
      <td>non-USA</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Volkswagen</td>
      <td>Eurovan</td>
      <td>Van</td>
      <td>16.6</td>
      <td>19.7</td>
      <td>22.7</td>
      <td>17</td>
      <td>21</td>
      <td>Driver</td>
      <td>Front</td>
      <td>...</td>
      <td>21.1</td>
      <td>7</td>
      <td>187</td>
      <td>115</td>
      <td>72</td>
      <td>38</td>
      <td>34.0</td>
      <td>14.0</td>
      <td>3960</td>
      <td>non-USA</td>
    </tr>
    <tr>
      <th>89</th>
      <td>Volkswagen</td>
      <td>Passat</td>
      <td>Compact</td>
      <td>17.6</td>
      <td>20.0</td>
      <td>22.4</td>
      <td>21</td>
      <td>30</td>
      <td>Driver</td>
      <td>Front</td>
      <td>...</td>
      <td>18.5</td>
      <td>5</td>
      <td>180</td>
      <td>103</td>
      <td>67</td>
      <td>35</td>
      <td>31.5</td>
      <td>14.0</td>
      <td>2985</td>
      <td>non-USA</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Volkswagen</td>
      <td>Corrado</td>
      <td>Sporty</td>
      <td>22.9</td>
      <td>23.3</td>
      <td>23.7</td>
      <td>18</td>
      <td>25</td>
      <td>Driver</td>
      <td>Front</td>
      <td>...</td>
      <td>18.5</td>
      <td>4</td>
      <td>159</td>
      <td>97</td>
      <td>66</td>
      <td>36</td>
      <td>26.0</td>
      <td>15.0</td>
      <td>2810</td>
      <td>non-USA</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Volvo</td>
      <td>240</td>
      <td>Compact</td>
      <td>21.8</td>
      <td>22.7</td>
      <td>23.5</td>
      <td>21</td>
      <td>28</td>
      <td>Driver</td>
      <td>Rear</td>
      <td>...</td>
      <td>15.8</td>
      <td>5</td>
      <td>190</td>
      <td>104</td>
      <td>67</td>
      <td>37</td>
      <td>29.5</td>
      <td>14.0</td>
      <td>2985</td>
      <td>non-USA</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Volvo</td>
      <td>850</td>
      <td>Midsize</td>
      <td>24.8</td>
      <td>26.7</td>
      <td>28.5</td>
      <td>20</td>
      <td>28</td>
      <td>Both</td>
      <td>Front</td>
      <td>...</td>
      <td>19.3</td>
      <td>5</td>
      <td>184</td>
      <td>105</td>
      <td>69</td>
      <td>38</td>
      <td>30.0</td>
      <td>15.0</td>
      <td>3245</td>
      <td>non-USA</td>
    </tr>
  </tbody>
</table>
