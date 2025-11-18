import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.width', 500) 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
loans = pd.read_csv('loan_data.csv')
#print(df.head())
#print(df.info())
#print(df.describe())
sns.set_style('darkgrid')
# For Credit Policy
'''
df1=df[df['credit.policy']==1]
df2=df[df['credit.policy']==0]
fig,ax = plt.subplots(figsize=(8, 6))
sns.histplot(df1['fico'],color='blue',edgecolor="black",lw=0.4,alpha=0.6,label="Credit Policy = 1", ax=ax,bins=30)
sns.histplot(df2['fico'],color='red',edgecolor="black",lw=0.4,alpha=0.6,label="Credit Policy = 0", ax=ax,bins=30)
ax.legend()
plt.show()
#For Not Fully Paid
df1=df[df['not.fully.paid']==1]
df2=df[df['not.fully.paid']==0]
fig,ax = plt.subplots(figsize=(8, 6))
sns.histplot(df1['fico'],color='blue',edgecolor="black",lw=0.4,alpha=0.6,label="Not Fully Paid= 1", ax=ax,bins=30)
sns.histplot(df2['fico'],color='red',edgecolor="black",lw=0.4,alpha=0.6,label="Not Fully Paid = 0", ax=ax,bins=30)
ax.legend()
plt.show()
sns.countplot(x='purpose',hue='not.fully.paid',data=df)
sns.jointplot(x='fico',y='int.rate',data=df,color='purple')
sns.lmplot(x='fico',y='int.rate',hue='credit.policy',col='not.fully.paid',data=df)
plt.show()'''

cat_feats=['purpose']
loan_dummies=pd.get_dummies(loans,columns=cat_feats,drop_first=True)
print(loan_dummies.iloc[0])
#loans.drop(['purpose'],axis=1,inplace=True)
#loans=pd.concat([loans,loan_dummies],axis=1)
loan_dummies[['purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational','purpose_home_improvement','purpose_major_purchase','purpose_small_business']] = loan_dummies[['purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational','purpose_home_improvement','purpose_major_purchase','purpose_small_business']].astype(int)
print(loan_dummies.iloc[0])


from sklearn.model_selection import train_test_split
X = loan_dummies.drop('not.fully.paid',axis=1)
y = loan_dummies['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print("FOR DECISION TREE \n")
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=300, random_state=42)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print("FOR RANDOM FOREST \n")
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))






























































