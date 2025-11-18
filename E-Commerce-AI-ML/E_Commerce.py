import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.width', 500) 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
sns.set_style('darkgrid')
df=pd.read_csv('Ecommerce Customers')
#print(df.head())
#print(df.info())
#print(df.describe())
#sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=df,edgecolor='black',color='#00FF19',lw=0.5,alpha=1)
#sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=df,edgecolor='black',color='red',lw=0.5,alpha=1)
#sns.jointplot(x='Time on App',y='Length of Membership',data=df,color='blue',kind='hex')
#sns.pairplot(df)
#sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=df,scatter_kws={'s':20,'color':'red','edgecolor':'black','linewidths':0.5},line_kws={'color':'black'})
#print(df.columns)
X=df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
Y=df[['Yearly Amount Spent']]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)
#print(X.head())
#print(lm.coef_)
predictions=lm.predict(X_test)
#plt.scatter(Y_test,predictions,edgecolor='black',color='blue',lw=0.5)
#plt.show()
from sklearn import metrics
#print("MAE= {}".format(metrics.mean_absolute_error(Y_test,predictions)))
#print("MSE= {}".format(metrics.mean_squared_error(Y_test,predictions)))
#print("RMSE= {}".format(np.sqrt(metrics.mean_squared_error(Y_test,predictions))))
#sns.histplot((Y_test-predictions),edgecolor='black',color='green',lw=0.5,bins=40)
coeff_df = pd.DataFrame(lm.coef_.flatten(),X.columns,columns=['Coefficient'])
print(coeff_df)
#plt.show()
