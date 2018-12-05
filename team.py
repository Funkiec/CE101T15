#pandas = Python Data Analysis Library
import pandas as pd

import matplotlib.pyplot as plt

#statistical data visualization
import seaborn as sns


df_train = pd.read_csv('C:/Users/Vish\Desktop/train.csv')
print(df_train.columns)
sns.distplot(df_train['SalePrice'])

#scatterplot between two variables
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

#scatter between most
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show()