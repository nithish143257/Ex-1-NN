<H3>ENTER YOUR NAME: NITHISH KUMAR P</H3>
<H3>ENTER YOUR REGISTER NO: 212221040115</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 14.02.2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df
#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
#Checking for null values
df.isnull().sum()
#Checking for duplicate values
df.duplicated()
#Describing the dataset
df.describe()
#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))


```


## OUTPUT:
### The Dataset:
![image](https://github.com/nithish143257/Ex-1-NN/assets/113762839/c64cf59c-3337-4e72-baf3-a7971bb763d8)
### Dropping unwanted features
![image](https://github.com/nithish143257/Ex-1-NN/assets/113762839/95b0e46e-2f11-4c11-9c04-2a0611ba4a3d)
### Checking for null values
![image](https://github.com/nithish143257/Ex-1-NN/assets/113762839/04383aee-13c4-4b99-856f-e3406774837e)
### Checking for duplication
![image](https://github.com/nithish143257/Ex-1-NN/assets/113762839/19b2247f-1dd3-4870-a58c-dae2c6032797)
### Describing the dataset
![image](https://github.com/nithish143257/Ex-1-NN/assets/113762839/cce905a0-9214-4d0d-96a8-7f8eb4a54a51)
### Scaling the values
![image](https://github.com/nithish143257/Ex-1-NN/assets/113762839/c85d9637-07c5-4e7f-9fcc-efc532f213c3)
### X Features
![image](https://github.com/nithish143257/Ex-1-NN/assets/113762839/ed9df820-b7b4-498e-9faa-5eb74bf03bbc)
### Y Features
![image](https://github.com/nithish143257/Ex-1-NN/assets/113762839/2c0d4f55-3fb0-4c96-91f8-320e9aa8c6f3)
### Splitting the training and testing dataset
![image](https://github.com/nithish143257/Ex-1-NN/assets/113762839/8f5abe47-1525-492b-bd78-f6378c272c44)









## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


