#Used for data analysis
import pandas as pd
#Used for generating graphs
import matplotlib.pyplot as plt
#Used ontop of matplotlib to make more complex visualizations
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('../input/train.csv')


#correlation matrix
cormap = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(cormap, vmax=.8, square=True);

#saleprice correlation matrix
numOfVars = 10 #number of variables for heatmap
cols = corrmat.nlargest(numOfVars, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
