# Most Frequently used EDA Codes

### import the warnings.

```
import warnings
warnings.filterwarnings("ignore")
```


### import the useful libraries.

```

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


### Read CSV FILE
```

df= pd.read_csv("https://raw.githubusercontent.com/datagrad/EDA-of-Bank-Telemarketing-Campaign/main/bank_marketing_updated_v1.csv?token=GHSAT0AAAAAABT3FJQMBRCCFK2Z5F4BT3TKYTTSUHA")
```


### Print the head of the data without first 2 rows.
```
df= pd.read_csv("https://raw.githubusercontent.com/datagrad/EDA-of-Bank-Telemarketing-Campaign/main/bank_marketing_updated_v1.csv?token=GHSAT0AAAAAABT3FJQMBRCCFK2Z5F4BT3TKYTTSUHA",skiprows=2)
df.head()
```



### Head and Tail of Dataset
```
df.head().append(df.tail())
```

### Check List of all Columns:
```

df.columns
```

### Shape: Rows and Columns
```
df.shape
```

### Info for Columns, Data Type, and Entries
```
df.info()
```

### Check Duplicate Rows count

```
df.duplicated().sum()
```

### Drop Duplicate Rows
```

df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
```

### Null Count
```

df.isnull().sum()
```

### Null Percentage

```
percent = 100*(df.isnull().sum())/int(df.shape[0])
percent.sort_values(ascending=False)
```

### Number of Rows with Null Value Count

```
df.isnull().any(axis = 1).sum()
```



### Analyzing the rows with null values

```
df.loc[df.isnull().any(axis=1)]
```


### Analyzing the rows where one column has null values to check pattern

```
df.loc[df['age'].isnull()]
```


### Numerical Column Description
```
df.describe()
```

### Value Counts 

```
df.month.value_counts()
```

### Mode Imputation

```
df['month'] = df['month'].fillna(df['month'].mode()[0])
df['month']
df.month.value_counts()
```


### Specific Value Imputation

```
df['month'] = df['month'].fillna('Missing or Unknown')
df['month']
df.month.value_counts()
```


### Drop all rows with Null Value in table

```
df = df.dropna()
```

### Drop Null value rows of one Column

```
df = df[~ df.age.isnull()]
```

### Drop a Column

```
df.drop("customerid", axis = 1, inplace = True)
```

### convert the data type from float to integer

```
df['age'] = df['age'].astype(int)
```


### Mean, Mode, Median

```
df['age'].aggregate(['mean','median'])
```
```

df['age'].mean()
```
```

df['age'].median()
```
```

df['age'].mode()[0]
```


### quantile

```
df.balance.quantile([.5,.7,.9, .95, .99])
```


### Data Capping

```
df[df.balance > 15000]. describe()
```

### Split column

```
df['job']= df.jobedu.apply(lambda x: x.split(",")[0])
```


### Replace with NAN (here wanted to replace all negative with Null Value)

```
df.loc[df.pdays < 0, "pdays"] = np.NaN
```

### Change data from one Unit to Another Unit (Minute to Second in Code)

#### Conversion of all data in column to a single standards. For example: if data are in minutes and seconds, then converting all to either second or minutes

##### convert the duration variable into single unit i.e. Seconds. and remove the sec or min prefix.

```
df.duration =df.duration.apply(
    lambda x: 
    float(x.split()[0])*60 if x.find("min")>0
    else float(x.split()[0])
    )
```




### Split a column on Delimiter 


##### Extract job in newly created 'job' column from "jobedu" column.

```
df['job']= df.jobedu.apply(lambda x: x.split(",")[0])
```


## Univariate Analysis & Data Quality Check for Categorical

#### Frequency Count
```
df.job.value_counts()
```

#### Column Description
```
df.job.describe()
```

#### Percentage Count
```
df.job.value_counts(normalize = True)
```

#### Bar Plot
```
df.job.value_counts().plot.bar()
```

#### Horizontal Bar Plot
```
df.job.value_counts().plot.barh()
```

## Univariate Analysis & Data Quality Check for Numerical

```
df.balance.describe()
```

#### Histogram (For Numerical)
```
df.balance.plot.hist()
```

#### Pie Chart
```
df.job.value_counts().plot.pie()
```

#### Percentile / Quantile

```
df.salary.quantile([.25,.50,.60,.75,.90])
```

#### Box Plot
```
sns.boxplot(df.salary)
```

## MultiVariate Analysis

**Correlation doesn't imply causation**


* Scatter Plot - for 2 Numerical variables.

* Pairplot - For all aor a set of numerical variables.

* Correlation -  To find numerical correlation between numerical variables (for one or all)

* Heatmap - to identify and plot the correlation with color density ranging between -1 to 1

#### Scatter Plot

```
plt.scatter(df.salary, df.balance)
```

#### Pair Plot
```
sns.pairplot(df)
```

#### Pair Plot for a set of Variables
```
sns.pairplot(df, vars = ["salary", "balance","age", "job"])
```



### Correlation Matrix used for only numerical to numerical

##### Create correlation Matrix

```
df.corr()
```

##### Use correlation Matrix to plot Heatmap

```
plt.figure(figsize = [10,10])
sns.heatmap(df.corr(),annot=True,cmap='Reds')
```


### Heatmap of Variables

##### plot the correlation matrix of salary, balance and age in inp1 dataframe.

```
sns.heatmap(df[['salary',"age", "balance"]].corr(),annot = True,cmap ='Reds')
plt.show()
```

### Numerical - Categorical

### Check Grouped Column

```
df.groupby("response")["salary"].mean()
```
```

df.groupby("response")["salary"].median()
```
```

df.groupby('response')['balance'].describe()
```

### Percentile function to be used as per requirement

```
def p25(x):
  return np.quantile(x,0.25)
def p50(x):
  return np.quantile(x,0.5)
def p75(x):
  return np.quantile(x,0.75)
```


### Using Percentile function

```
df.groupby('response')['balance'].aggregate(['mean',p25,'median', p75])
```


### BOX Plot

```
sns.boxplot(data=df, x='response', y='salary')
```



### Bar Plot with mean and Median

```
df.groupby('response')['balance'].aggregate(['mean','median']).plot.bar()
```

### Bar Plot with Mean, Median, and Quantile

```
df.groupby('response')['balance'].aggregate(['mean',p25,'median', p75]).plot.bar()
```

### Categorical - Categorical Multivariate


### Flagging the target Variable

```
df['resp_flag'] = np.where(df.response == 'yes',1,0)
```

```
df.resp_flag.value_counts()
```


### Understanding Flag Variable

```
df.resp_flag.describe()
```

### Mean & Sum of Flag Variable

```
df.resp_flag.mean()
```


```
df.groupby(['marital','loan'])['resp_flag'].mean()
```


#### Bar Plot Multivariate with Mean Comapre

```
df.groupby(['marital','loan'])['resp_flag'].mean().plot.bar()
```

#### Bar Plot Multivariate with Mean, median and Quantile Comapre

##### Percentile function to be used as per requirement

###### Create Quantile

```
def p25(x):
  return np.quantile(x,0.25)
def p50(x):
  return np.quantile(x,0.5)
def p75(x):
  return np.quantile(x,0.75)
```

###### Plot Bar Graph

```
df.groupby(['marital','loan'])['resp_flag'].aggregate([p25, p50, p75, 'mean']).plot.bar()
```
```

df.groupby(['marital','loan'])['resp_flag'].aggregate([p25, p50, p75, 'mean'])
```

## Binning


### Binning

```
df['age_bins'] = pd.cut(
    df.age,
    [0,30,40,50,60,999],
    labels = ['<30','30-40','40-50','50-60','60+']
)
```

### Value Count in Bins

```
df.age_bins.value_counts(normalize = True)
```

### Bar Plot on Bins

```
df.groupby(['age_bins'])['response'].value_counts().plot.bar()
```

### bar plot to compare Bins

```
plt.figure(figsize = [10,4])
plt.subplot(1,2,1)
df.age_bins.value_counts(normalize = True).plot.bar()
plt.subplot(1,2,2)
df.groupby(['age_bins'])['resp_flag'].mean().plot.bar()
```



## Pivot Table & HeatMap for Categorical to Categorical

### Create Pivot Table

```
job_age = pd.pivot_table(data = df,
                             index = 'job',
                             columns = ['age_bins'],
                             values = 'resp_flag'
                             )
```

### Use Pivot table to Plot HeatMap

```
sns.heatmap(job_age,
            annot = True,
            cmap = 'RdYlGn',
            center = 0.117    
)
```
