import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.width', 500) 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
sns.set_style('darkgrid')
df=pd.read_csv('USA_Housing.csv')
#sns.distplot(df['Price'],kde=True)
#sns.heatmap(df.corr(numeric_only=True), cmap='magma_r', annot=True)
#plt.title("Correlation Heatmap")
#plt.show()
X=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
Y=df['Price']
print(df.info())

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)
predictions=lm.predict(X_test)
#plt.scatter(Y_test,predictions,edgecolor='black',color='red',lw=0.5)
sns.histplot((Y_test-predictions),edgecolor='black',color='red',lw=0.5,kde=True,alpha=1)
plt.show()

