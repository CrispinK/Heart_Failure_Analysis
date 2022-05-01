# HEART FAILURE DATA ANALYSIS

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide. Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.
People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

### IMPORTING LIBRARIES


```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import svm 
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from keras import callbacks
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
```

### LOADING DATA 


```python
Heart_Failure = pd.read_csv('heart_failure.csv')
Heart_Failure.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>anaemia</th>
      <th>creatinine_phosphokinase</th>
      <th>diabetes</th>
      <th>ejection_fraction</th>
      <th>high_blood_pressure</th>
      <th>platelets</th>
      <th>serum_creatinine</th>
      <th>serum_sodium</th>
      <th>sex</th>
      <th>smoking</th>
      <th>time</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75.0</td>
      <td>0</td>
      <td>582</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
      <td>265000.00</td>
      <td>1.9</td>
      <td>130</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.0</td>
      <td>0</td>
      <td>7861</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>263358.03</td>
      <td>1.1</td>
      <td>136</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65.0</td>
      <td>0</td>
      <td>146</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>162000.00</td>
      <td>1.3</td>
      <td>129</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.0</td>
      <td>1</td>
      <td>111</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>210000.00</td>
      <td>1.9</td>
      <td>137</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65.0</td>
      <td>1</td>
      <td>160</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>327000.00</td>
      <td>2.7</td>
      <td>116</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### About the data (Description of attributes)
* **age**: Age of the patient
* **anaemia**: Haemoglobin level of patient (Boolean)
* **creatinine_phosphokinase**: Level of the CPK enzyme in the blood (mcg/L)
* **diabetes**: If the patient has diabetes (Boolean)
* **ejection_fraction**: Percentage of blood leaving the heart at each contraction
* **high_blood_pressure**: If the patient has hypertension (Boolean)
* **platelets**: Platelet count of blood (kiloplatelets/mL)
* **serum_creatinine**: Level of serum creatinine in the blood (mg/dL)
* **serum_sodium**: Level of serum sodium in the blood (mEq/L)
* **sex**: Sex of the patient
* **smoking**: If the patient smokes or not (Boolean)
* **time**: Follow-up period (days)
* **DEATH_EVENT**: If the patient deceased during the follow-up period (Boolean)
* [Attributes having Boolean values: 0 = Negative (No); 1 = Positive (Yes)]

### DATA ANALYSIS

#### Task #1: Checking for any missing values across the dataset


```python
Heart_Failure.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 299 entries, 0 to 298
    Data columns (total 13 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   age                       299 non-null    float64
     1   anaemia                   299 non-null    int64  
     2   creatinine_phosphokinase  299 non-null    int64  
     3   diabetes                  299 non-null    int64  
     4   ejection_fraction         299 non-null    int64  
     5   high_blood_pressure       299 non-null    int64  
     6   platelets                 299 non-null    float64
     7   serum_creatinine          299 non-null    float64
     8   serum_sodium              299 non-null    int64  
     9   sex                       299 non-null    int64  
     10  smoking                   299 non-null    int64  
     11  time                      299 non-null    int64  
     12  DEATH_EVENT               299 non-null    int64  
    dtypes: float64(3), int64(10)
    memory usage: 30.5 KB
    

#### Note:
* There are 299 non-null values in all the attributes thus no missing values.
* Datatype is also either 'float64' or 'int64' which works well while feeded to an algorithm.

#### Task #2: Evaluating the target and finding out the potential skewness in the data 


```python
cols= ["#CD5C5C","#FF0000"]
ax = sns.countplot(x= Heart_Failure["DEATH_EVENT"], palette= cols)
ax.bar_label(ax.containers[0])
```




    [Text(0, 0, '203'), Text(0, 0, '96')]





    
![output_12_1](https://user-images.githubusercontent.com/75635908/166144233-19490941-551d-4faf-9bb5-37bcd9335605.png)
    


#### Note:
* Target labels are 203 versus 96 thus there is an imbalance in the data. 

#### Task #3: Doing Univariate Analysis for statistical description and understanding of dispersion of data 


```python
Heart_Failure.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>299.0</td>
      <td>60.833893</td>
      <td>11.894809</td>
      <td>40.0</td>
      <td>51.0</td>
      <td>60.0</td>
      <td>70.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>anaemia</th>
      <td>299.0</td>
      <td>0.431438</td>
      <td>0.496107</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>creatinine_phosphokinase</th>
      <td>299.0</td>
      <td>581.839465</td>
      <td>970.287881</td>
      <td>23.0</td>
      <td>116.5</td>
      <td>250.0</td>
      <td>582.0</td>
      <td>7861.0</td>
    </tr>
    <tr>
      <th>diabetes</th>
      <td>299.0</td>
      <td>0.418060</td>
      <td>0.494067</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ejection_fraction</th>
      <td>299.0</td>
      <td>38.083612</td>
      <td>11.834841</td>
      <td>14.0</td>
      <td>30.0</td>
      <td>38.0</td>
      <td>45.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>high_blood_pressure</th>
      <td>299.0</td>
      <td>0.351171</td>
      <td>0.478136</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>platelets</th>
      <td>299.0</td>
      <td>263358.029264</td>
      <td>97804.236869</td>
      <td>25100.0</td>
      <td>212500.0</td>
      <td>262000.0</td>
      <td>303500.0</td>
      <td>850000.0</td>
    </tr>
    <tr>
      <th>serum_creatinine</th>
      <td>299.0</td>
      <td>1.393880</td>
      <td>1.034510</td>
      <td>0.5</td>
      <td>0.9</td>
      <td>1.1</td>
      <td>1.4</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>serum_sodium</th>
      <td>299.0</td>
      <td>136.625418</td>
      <td>4.412477</td>
      <td>113.0</td>
      <td>134.0</td>
      <td>137.0</td>
      <td>140.0</td>
      <td>148.0</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>299.0</td>
      <td>0.648829</td>
      <td>0.478136</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>smoking</th>
      <td>299.0</td>
      <td>0.321070</td>
      <td>0.467670</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>time</th>
      <td>299.0</td>
      <td>130.260870</td>
      <td>77.614208</td>
      <td>4.0</td>
      <td>73.0</td>
      <td>115.0</td>
      <td>203.0</td>
      <td>285.0</td>
    </tr>
    <tr>
      <th>DEATH_EVENT</th>
      <td>299.0</td>
      <td>0.321070</td>
      <td>0.467670</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Note:
* Features "creatinine_phosphokinase" & "serum creatinine" are significantly skewed.
* All the other features almost shows the normal distribution, since mean is equal to their respective medians. 

#### Task #4: Doing Bivariate Analysis by examaning a corelation matrix of all the features using heatmap


```python
cmap = sns.diverging_palette(2, 165, s=80, l=55, n=9)
corrmat = Heart_Failure.corr()
plt.subplots(figsize=(20,20))
sns.heatmap(corrmat,cmap= cmap,annot=True, square=True)
```




    <AxesSubplot:>




    
![output_18_1](https://user-images.githubusercontent.com/75635908/166144383-a9c7e6e5-dbfb-49a5-a8a0-97567e8e64e4.png)
    



#### Note:
* "time" is the most important feature as it would've been very crucial to get diagnosed early with cardivascular issue so as to get timely treatment thus, reducing the chances of any fatality. (Evident from the inverse relationship)

* "serum_creatinine" is the next important feature as serum's (essential component of blood) abundancy in blood makes it easier for heart to function.

* "ejection_fraction" has also significant influence on target variable which is expected since it is basically the efficiency of the heart.

* Can be seen from the inverse relation pattern that heart's functioning declines with ageing.

#### Task #5: Evaluating age distribution as per the deaths happened


```python
plt.figure(figsize=(15,10))
Days_of_week=sns.countplot(x=Heart_Failure['age'],data=Heart_Failure, hue ="DEATH_EVENT",palette = cols)
Days_of_week.set_title("Distribution Of Age", color="#774571")
```




    Text(0.5, 1.0, 'Distribution Of Age')




    
![output_21_1](https://user-images.githubusercontent.com/75635908/166144416-0bfdc45b-2dbf-4c50-a691-f4a562e35156.png)

    


#### Task #6: Checking for potential outliers using the "Boxen and Swarm plots" of non binary features.


```python
feature = ["age","creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium", "time"]
for i in feature: 
    plt.figure(figsize=(10,7))
    sns.swarmplot(x=Heart_Failure["DEATH_EVENT"], y=Heart_Failure[i], color="black", alpha=0.7)
    sns.boxenplot(x=Heart_Failure["DEATH_EVENT"], y=Heart_Failure[i], palette=cols)
    plt.show() 
```


    
![output_23_0](https://user-images.githubusercontent.com/75635908/166144453-e7d4c276-e873-424f-a6c5-670894912ac7.png)

    



    
![output_23_1](https://user-images.githubusercontent.com/75635908/166144492-06b94646-e35a-4379-b263-7c1f72ef63ba.png)




    
![output_23_2](https://user-images.githubusercontent.com/75635908/166144494-d8824ee3-28a5-40cc-af61-052e8bf4d6b4.png)


    
![output_23_3](https://user-images.githubusercontent.com/75635908/166144520-60e7dcf3-3776-4e83-b883-07c4b7fa595b.png)

    
![output_23_4](https://user-images.githubusercontent.com/75635908/166144536-9b2c6203-5398-41cd-a921-11d381e0ba20.png)

    



    
![output_23_5](https://user-images.githubusercontent.com/75635908/166144559-f76e9690-2d4b-4342-a371-fe0ae8a44438.png)
    



    
![output_23_6](https://user-images.githubusercontent.com/75635908/166144587-ab6d74c9-93c9-4a9f-87df-221844920233.png)

    



#### Note:
* Few Outliers can be seen in almost all the features
* Considering the size of the dataset and relevancy of it, we won't be dropping such outliers in data preprocessing which wouldn't bring any statistical fluke.

#### Task #7: Plotting "Kernel Density Estimation (kde plot)" of time and age features -  both of which are significant ones.


```python
sns.kdeplot(x=Heart_Failure["time"], y=Heart_Failure["age"], hue =Heart_Failure["DEATH_EVENT"], palette=cols)
```




    <AxesSubplot:xlabel='time', ylabel='age'>




    
![output_26_1](https://user-images.githubusercontent.com/75635908/166144625-d93b8001-da08-4596-8424-5b73aaf4aab3.png)

    


#### Note:
* With less follow-up days, patients often died only when they aged more.
* More the follow-up days more the probability is less of any fatality.

### DATA PREPROCESSING

#### Task #8: Defining independent and dependent attributes in training and test sets


```python
X=Heart_Failure.drop(["DEATH_EVENT"],axis=1)
y=Heart_Failure["DEATH_EVENT"]
```

#### Task #9: Setting up a standard scaler for the features and analyzing it thereafter


```python
col_names = list(X.columns)
s_scaler = preprocessing.StandardScaler()
X_scaled= s_scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=col_names)   
X_scaled.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>299.0</td>
      <td>5.703353e-16</td>
      <td>1.001676</td>
      <td>-1.754448</td>
      <td>-0.828124</td>
      <td>-0.070223</td>
      <td>0.771889</td>
      <td>2.877170</td>
    </tr>
    <tr>
      <th>anaemia</th>
      <td>299.0</td>
      <td>1.009969e-16</td>
      <td>1.001676</td>
      <td>-0.871105</td>
      <td>-0.871105</td>
      <td>-0.871105</td>
      <td>1.147968</td>
      <td>1.147968</td>
    </tr>
    <tr>
      <th>creatinine_phosphokinase</th>
      <td>299.0</td>
      <td>0.000000e+00</td>
      <td>1.001676</td>
      <td>-0.576918</td>
      <td>-0.480393</td>
      <td>-0.342574</td>
      <td>0.000166</td>
      <td>7.514640</td>
    </tr>
    <tr>
      <th>diabetes</th>
      <td>299.0</td>
      <td>9.060014e-17</td>
      <td>1.001676</td>
      <td>-0.847579</td>
      <td>-0.847579</td>
      <td>-0.847579</td>
      <td>1.179830</td>
      <td>1.179830</td>
    </tr>
    <tr>
      <th>ejection_fraction</th>
      <td>299.0</td>
      <td>-3.267546e-17</td>
      <td>1.001676</td>
      <td>-2.038387</td>
      <td>-0.684180</td>
      <td>-0.007077</td>
      <td>0.585389</td>
      <td>3.547716</td>
    </tr>
    <tr>
      <th>high_blood_pressure</th>
      <td>299.0</td>
      <td>0.000000e+00</td>
      <td>1.001676</td>
      <td>-0.735688</td>
      <td>-0.735688</td>
      <td>-0.735688</td>
      <td>1.359272</td>
      <td>1.359272</td>
    </tr>
    <tr>
      <th>platelets</th>
      <td>299.0</td>
      <td>7.723291e-17</td>
      <td>1.001676</td>
      <td>-2.440155</td>
      <td>-0.520870</td>
      <td>-0.013908</td>
      <td>0.411120</td>
      <td>6.008180</td>
    </tr>
    <tr>
      <th>serum_creatinine</th>
      <td>299.0</td>
      <td>1.425838e-16</td>
      <td>1.001676</td>
      <td>-0.865509</td>
      <td>-0.478205</td>
      <td>-0.284552</td>
      <td>0.005926</td>
      <td>7.752020</td>
    </tr>
    <tr>
      <th>serum_sodium</th>
      <td>299.0</td>
      <td>-8.673849e-16</td>
      <td>1.001676</td>
      <td>-5.363206</td>
      <td>-0.595996</td>
      <td>0.085034</td>
      <td>0.766064</td>
      <td>2.582144</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>299.0</td>
      <td>-8.911489e-18</td>
      <td>1.001676</td>
      <td>-1.359272</td>
      <td>-1.359272</td>
      <td>0.735688</td>
      <td>0.735688</td>
      <td>0.735688</td>
    </tr>
    <tr>
      <th>smoking</th>
      <td>299.0</td>
      <td>-1.188199e-17</td>
      <td>1.001676</td>
      <td>-0.687682</td>
      <td>-0.687682</td>
      <td>-0.687682</td>
      <td>1.454161</td>
      <td>1.454161</td>
    </tr>
    <tr>
      <th>time</th>
      <td>299.0</td>
      <td>-1.901118e-16</td>
      <td>1.001676</td>
      <td>-1.629502</td>
      <td>-0.739000</td>
      <td>-0.196954</td>
      <td>0.938759</td>
      <td>1.997038</td>
    </tr>
  </tbody>
</table>
</div>



#### Task #10: Plotting the scaled features using boxen plots


```python
colors =["#CD5C5C","#F08080","#FA8072","#E9967A","#FFA07A"]
plt.figure(figsize=(20,10))
sns.boxenplot(data = X_scaled,palette = colors)
plt.xticks(rotation=60)
plt.show()
```


    
![output_34_0](https://user-images.githubusercontent.com/75635908/166144646-592704b8-322e-418a-a9d9-0b4adce1caf5.png)

    


#### Task #11: spliting variables into training and test sets


```python
X_train, X_test, y_train,y_test = train_test_split(X_scaled,y,test_size=0.30,random_state=25)
```

### MODEL BUILDING
#### 1. SUPPORT VECTOR MACHINE (SVM)

#### Task #12: Instantiating the SVM algorithm, Fitting the model, Predicting the test variables and Getting the score


```python
model1=svm.SVC()

model1.fit (X_train, y_train)

y_pred = model1.predict(X_test)

model1.score (X_test, y_test)
```




    0.7888888888888889



#### Task #13: Printing classification report (since there was biasness in target labels)


```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.84      0.85      0.84        60
               1       0.69      0.67      0.68        30
    
        accuracy                           0.79        90
       macro avg       0.76      0.76      0.76        90
    weighted avg       0.79      0.79      0.79        90
    
    

#### Task #14: Getting the confusion matrix


```python
cmap1 = sns.diverging_palette(2, 165, s=80, l=55, n=9)
plt.subplots(figsize=(10,7))
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap = cmap1, annot = True, annot_kws = {'size':25})
```




    <AxesSubplot:>




    
![output_43_1](https://user-images.githubusercontent.com/75635908/166144668-9db39198-f159-46d8-9e0d-02e819cd1006.png)



#### 2. Artificial Neural Network (ANN) 


```python
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True)

# Initialising the NN
model = Sequential()

# layers
model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 12))
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
history = model.fit(X_train, y_train, batch_size = 25, epochs = 80,callbacks=[early_stopping], validation_split=0.25)
```

    Epoch 1/80
    7/7 [==============================] - 1s 33ms/step - loss: 0.6928 - accuracy: 0.6410 - val_loss: 0.6908 - val_accuracy: 0.8302
    Epoch 2/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.6918 - accuracy: 0.6346 - val_loss: 0.6885 - val_accuracy: 0.8302
    Epoch 3/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.6909 - accuracy: 0.6346 - val_loss: 0.6863 - val_accuracy: 0.8302
    Epoch 4/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.6901 - accuracy: 0.6346 - val_loss: 0.6843 - val_accuracy: 0.8302
    Epoch 5/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.6892 - accuracy: 0.6346 - val_loss: 0.6823 - val_accuracy: 0.8302
    Epoch 6/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.6884 - accuracy: 0.6346 - val_loss: 0.6800 - val_accuracy: 0.8302
    Epoch 7/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.6873 - accuracy: 0.6346 - val_loss: 0.6772 - val_accuracy: 0.8302
    Epoch 8/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.6860 - accuracy: 0.6346 - val_loss: 0.6742 - val_accuracy: 0.8302
    Epoch 9/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.6845 - accuracy: 0.6346 - val_loss: 0.6712 - val_accuracy: 0.8302
    Epoch 10/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.6823 - accuracy: 0.6346 - val_loss: 0.6670 - val_accuracy: 0.8302
    Epoch 11/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.6805 - accuracy: 0.6346 - val_loss: 0.6616 - val_accuracy: 0.8302
    Epoch 12/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.6760 - accuracy: 0.6346 - val_loss: 0.6533 - val_accuracy: 0.8302
    Epoch 13/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.6710 - accuracy: 0.6346 - val_loss: 0.6422 - val_accuracy: 0.8302
    Epoch 14/80
    7/7 [==============================] - 0s 9ms/step - loss: 0.6619 - accuracy: 0.6346 - val_loss: 0.6277 - val_accuracy: 0.8302
    Epoch 15/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.6482 - accuracy: 0.6346 - val_loss: 0.6073 - val_accuracy: 0.8302
    Epoch 16/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.6352 - accuracy: 0.6346 - val_loss: 0.5811 - val_accuracy: 0.8302
    Epoch 17/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.6156 - accuracy: 0.6346 - val_loss: 0.5487 - val_accuracy: 0.8302
    Epoch 18/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.6024 - accuracy: 0.6346 - val_loss: 0.5128 - val_accuracy: 0.8302
    Epoch 19/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.5714 - accuracy: 0.6346 - val_loss: 0.4776 - val_accuracy: 0.8302
    Epoch 20/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.5533 - accuracy: 0.6346 - val_loss: 0.4429 - val_accuracy: 0.8302
    Epoch 21/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.5448 - accuracy: 0.6346 - val_loss: 0.4079 - val_accuracy: 0.8302
    Epoch 22/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.5098 - accuracy: 0.6346 - val_loss: 0.3860 - val_accuracy: 0.8302
    Epoch 23/80
    7/7 [==============================] - 0s 6ms/step - loss: 0.5223 - accuracy: 0.6346 - val_loss: 0.3647 - val_accuracy: 0.8302
    Epoch 24/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.5233 - accuracy: 0.6346 - val_loss: 0.3509 - val_accuracy: 0.8302
    Epoch 25/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.5034 - accuracy: 0.6346 - val_loss: 0.3438 - val_accuracy: 0.8302
    Epoch 26/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.4793 - accuracy: 0.6346 - val_loss: 0.3324 - val_accuracy: 0.8302
    Epoch 27/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.5073 - accuracy: 0.6346 - val_loss: 0.3219 - val_accuracy: 0.8302
    Epoch 28/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.4910 - accuracy: 0.6346 - val_loss: 0.3147 - val_accuracy: 0.8302
    Epoch 29/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.5124 - accuracy: 0.6346 - val_loss: 0.3113 - val_accuracy: 0.8302
    Epoch 30/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.5133 - accuracy: 0.6346 - val_loss: 0.3111 - val_accuracy: 0.8302
    Epoch 31/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4766 - accuracy: 0.6346 - val_loss: 0.3088 - val_accuracy: 0.8302
    Epoch 32/80
    7/7 [==============================] - 0s 9ms/step - loss: 0.4975 - accuracy: 0.6346 - val_loss: 0.3080 - val_accuracy: 0.8302
    Epoch 33/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.5191 - accuracy: 0.6346 - val_loss: 0.3073 - val_accuracy: 0.8302
    Epoch 34/80
    7/7 [==============================] - 0s 6ms/step - loss: 0.4627 - accuracy: 0.6346 - val_loss: 0.3029 - val_accuracy: 0.8302
    Epoch 35/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4563 - accuracy: 0.6346 - val_loss: 0.2953 - val_accuracy: 0.8302
    Epoch 36/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.4848 - accuracy: 0.6346 - val_loss: 0.2915 - val_accuracy: 0.8302
    Epoch 37/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.4772 - accuracy: 0.6346 - val_loss: 0.2920 - val_accuracy: 0.8302
    Epoch 38/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4652 - accuracy: 0.6346 - val_loss: 0.2912 - val_accuracy: 0.8302
    Epoch 39/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4511 - accuracy: 0.6346 - val_loss: 0.2884 - val_accuracy: 0.8302
    Epoch 40/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.4453 - accuracy: 0.6346 - val_loss: 0.2868 - val_accuracy: 0.8302
    Epoch 41/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4740 - accuracy: 0.6346 - val_loss: 0.2862 - val_accuracy: 0.8302
    Epoch 42/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.4492 - accuracy: 0.6346 - val_loss: 0.2844 - val_accuracy: 0.8302
    Epoch 43/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.4367 - accuracy: 0.6346 - val_loss: 0.2828 - val_accuracy: 0.8302
    Epoch 44/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4651 - accuracy: 0.6346 - val_loss: 0.2812 - val_accuracy: 0.8302
    Epoch 45/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4550 - accuracy: 0.6346 - val_loss: 0.2799 - val_accuracy: 0.8302
    Epoch 46/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.4571 - accuracy: 0.6346 - val_loss: 0.2819 - val_accuracy: 0.8868
    Epoch 47/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4477 - accuracy: 0.7564 - val_loss: 0.2817 - val_accuracy: 0.8868
    Epoch 48/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4479 - accuracy: 0.7756 - val_loss: 0.2814 - val_accuracy: 0.8868
    Epoch 49/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4063 - accuracy: 0.7949 - val_loss: 0.2817 - val_accuracy: 0.8868
    Epoch 50/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4578 - accuracy: 0.8205 - val_loss: 0.2802 - val_accuracy: 0.8868
    Epoch 51/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4499 - accuracy: 0.8205 - val_loss: 0.2792 - val_accuracy: 0.8868
    Epoch 52/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4375 - accuracy: 0.7949 - val_loss: 0.2771 - val_accuracy: 0.8868
    Epoch 53/80
    7/7 [==============================] - 0s 9ms/step - loss: 0.4340 - accuracy: 0.8333 - val_loss: 0.2761 - val_accuracy: 0.8868
    Epoch 54/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4182 - accuracy: 0.8333 - val_loss: 0.2745 - val_accuracy: 0.8868
    Epoch 55/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4072 - accuracy: 0.8333 - val_loss: 0.2755 - val_accuracy: 0.8868
    Epoch 56/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4231 - accuracy: 0.8333 - val_loss: 0.2741 - val_accuracy: 0.8868
    Epoch 57/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4398 - accuracy: 0.8397 - val_loss: 0.2732 - val_accuracy: 0.8868
    Epoch 58/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4172 - accuracy: 0.8654 - val_loss: 0.2722 - val_accuracy: 0.8868
    Epoch 59/80
    7/7 [==============================] - 0s 6ms/step - loss: 0.4264 - accuracy: 0.8397 - val_loss: 0.2722 - val_accuracy: 0.8868
    Epoch 60/80
    7/7 [==============================] - 0s 6ms/step - loss: 0.4185 - accuracy: 0.8397 - val_loss: 0.2724 - val_accuracy: 0.8868
    Epoch 61/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4107 - accuracy: 0.8397 - val_loss: 0.2743 - val_accuracy: 0.8868
    Epoch 62/80
    7/7 [==============================] - 0s 6ms/step - loss: 0.4347 - accuracy: 0.8526 - val_loss: 0.2762 - val_accuracy: 0.8679
    Epoch 63/80
    7/7 [==============================] - 0s 6ms/step - loss: 0.4120 - accuracy: 0.8397 - val_loss: 0.2782 - val_accuracy: 0.8679
    Epoch 64/80
    7/7 [==============================] - 0s 6ms/step - loss: 0.4141 - accuracy: 0.8462 - val_loss: 0.2798 - val_accuracy: 0.8679
    Epoch 65/80
    7/7 [==============================] - 0s 6ms/step - loss: 0.3953 - accuracy: 0.8654 - val_loss: 0.2792 - val_accuracy: 0.8679
    Epoch 66/80
    7/7 [==============================] - 0s 6ms/step - loss: 0.4200 - accuracy: 0.8654 - val_loss: 0.2790 - val_accuracy: 0.8679
    Epoch 67/80
    7/7 [==============================] - 0s 6ms/step - loss: 0.4119 - accuracy: 0.8654 - val_loss: 0.2845 - val_accuracy: 0.8679
    Epoch 68/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4106 - accuracy: 0.8462 - val_loss: 0.2856 - val_accuracy: 0.8679
    Epoch 69/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.3859 - accuracy: 0.8590 - val_loss: 0.2860 - val_accuracy: 0.8491
    Epoch 70/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.4077 - accuracy: 0.8654 - val_loss: 0.2886 - val_accuracy: 0.8491
    Epoch 71/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.3842 - accuracy: 0.8718 - val_loss: 0.2867 - val_accuracy: 0.8491
    Epoch 72/80
    7/7 [==============================] - 0s 11ms/step - loss: 0.3903 - accuracy: 0.8718 - val_loss: 0.2844 - val_accuracy: 0.8679
    Epoch 73/80
    7/7 [==============================] - 0s 6ms/step - loss: 0.3757 - accuracy: 0.8654 - val_loss: 0.2845 - val_accuracy: 0.8679
    Epoch 74/80
    7/7 [==============================] - 0s 7ms/step - loss: 0.3903 - accuracy: 0.8654 - val_loss: 0.2835 - val_accuracy: 0.8679
    Epoch 75/80
    7/7 [==============================] - 0s 8ms/step - loss: 0.3863 - accuracy: 0.8654 - val_loss: 0.2853 - val_accuracy: 0.8491
    Epoch 76/80
    7/7 [==============================] - 0s 6ms/step - loss: 0.3679 - accuracy: 0.8718 - val_loss: 0.2855 - val_accuracy: 0.8491
    Epoch 77/80
    7/7 [==============================] - 0s 6ms/step - loss: 0.3748 - accuracy: 0.8782 - val_loss: 0.2842 - val_accuracy: 0.8491
    Epoch 78/80
    7/7 [==============================] - 0s 6ms/step - loss: 0.4015 - accuracy: 0.8654 - val_loss: 0.2846 - val_accuracy: 0.8491
    


```python
val_accuracy = np.mean(history.history['val_accuracy'])
print("\n%s: %.2f%%" % ('val_accuracy is', val_accuracy*100))
```

    
    val_accuracy is: 84.83%
    


```python
history_df = pd.DataFrame(history.history)

plt.plot(history_df.loc[:, ['loss']], "#CD5C5C", label='Training loss')
plt.plot(history_df.loc[:, ['val_loss']],"#FF0000", label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")

plt.show()
```


    
![output_47_0](https://user-images.githubusercontent.com/75635908/166144690-bb04b767-f874-443a-85c3-83c65f0a41a4.png)

    



```python
history_df = pd.DataFrame(history.history)

plt.plot(history_df.loc[:, ['accuracy']], "#CD5C5C", label='Training accuracy')
plt.plot(history_df.loc[:, ['val_accuracy']],"#FF0000", label='Validation accuracy')

plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```


    
![output_48_0](https://user-images.githubusercontent.com/75635908/166144698-3a3c70ea-d902-4d66-a51c-f417b19ba342.png)

    


#### Task #15: Predicting the test set results


```python
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.4)
np.set_printoptions()
```

#### Task #16: Getting the confusion matrix


```python
cmap1 = sns.diverging_palette(2, 165, s=80, l=55, n=9)
plt.subplots(figsize=(10,7))
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap = cmap1, annot = True, annot_kws = {'size':25})
```




    <AxesSubplot:>




    
![output_52_1](https://user-images.githubusercontent.com/75635908/166144729-be091561-7b39-4b05-9f0c-1644a4bc13b1.png)

    



```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.84      0.78      0.81        60
               1       0.62      0.70      0.66        30
    
        accuracy                           0.76        90
       macro avg       0.73      0.74      0.73        90
    weighted avg       0.77      0.76      0.76        90
