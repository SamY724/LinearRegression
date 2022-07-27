import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression


#Reading dataset using pandas
df = pd.read_csv("ds_salaries.csv")

#dropping initial problematic rows
df = df.dropna(axis=0)

#First, the salary in usd is my Y variable here, which is what I want to predict, as such I am replacing any problematic
#entries with the mean salary

no_zero_values = ["salary_in_usd"]

for column in no_zero_values:
    df[column] = df[column].replace(0,np.NaN)
    #finds mean of the dataset of values that dont include 'none'
    mean = int(df[column].mean(skipna=True))
    # replace all of the 'none ' values with the means
    df[column] = df[column].replace(np.NaN,mean)




#altering important categorical data to allow for usage
df['company_size'] = df['company_size'].astype('category')
df['company_size_tier'] = df['company_size'].cat.codes


df['experience_level'] = df['experience_level'].astype('category')
df['experience_level_tier'] = df['experience_level'].cat.codes

salaries = df["salary_in_usd"]

count = salaries[salaries > 300000].count()

#as witnessed when printing the above count, only 10 salaries exceeded thre 300000 threshold, as such, they were dropped as outliers
df = df[df.salary_in_usd <= 300000]

#the below boxplot function confirms the lack of outliers in terms of salary now

def salaryUSDplot():
    sns.boxplot(x=df['salary_in_usd'])
    plt.show()


#removing job titles that do appear less than 8 times, this is largely ineffective for the dataset
max_repeat = 8
vc = df["job_title"].value_counts()
df = df[df["job_title"].isin(vc[vc > max_repeat].index)]

#Now to encode the job titles using OHE approach

encodedTitles = pd.get_dummies(df.job_title)

df = pd.concat([df,encodedTitles],axis=1)


#Dropping irrelevant columns
to_drop = ["Unnamed: 0","employment_type","experience_level","company_size","salary","job_title","salary_currency"
,"company_location","employee_residence","remote_ratio","job_title"]


df.drop(columns=to_drop,inplace=True)


#setting x variables (independent)
x = df.drop(["salary_in_usd"],axis=1)


#setting y variable (dependent)
y = df["salary_in_usd"]


#now to perform linear regression
lr = LinearRegression()

model = lr.fit(X=x,y=y)


print(f'alpha = {model.intercept_}')
print(f'betas = {model.coef_}')


print(model.predict(x))
