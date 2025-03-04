from pandas import value_counts
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

"""to read the csv file"""
df=pd.read_csv("Salary_dataset.csv")
print(df.head())

"""to know the datatype"""
column_data_types=df.dtypes
print(column_data_types)

"""Finding null values"""
print(df.isnull().sum())

"""to know the shape of dataset"""
# df_shape=df.shape
# print(df_shape)

"""to know information of dataset"""
# df_info=df.info
# print(df_info)

"""to know the describtion of the dataset"""
# print(df.describe())


"""Value count for yearsExperience"""
value_count_for=df['YearsExperience'].value_counts()
print("Value Count:",value_count_for)

"""pyplot"""
fig,axes=plt.subplots(3,1,figsize=(5,5))
axes[0].boxplot(df["YearsExperience"])
axes[1].boxplot(df["Salary"])
plt.tight_layout()
plt.show()

"""Pairplot"""
sns.pairplot(df,x_vars="YearsExperience",y_vars="Salary",height=4,aspect=1,kind="scatter")
plt.show()

"""Heatmap"""
sns.heatmap(df.corr(),cmap="magma",annot=True)
plt.show()

"""Train and Test of Data"""
x=df[['YearsExperience']]
y=df['Salary']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
output=model.predict(x_test)
print(output)

"""Testing Accuary"""
testing_accuracy=r2_score(y_test,output)
print("Accuracy",testing_accuracy)







