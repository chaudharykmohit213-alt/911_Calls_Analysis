import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cancer_classification.csv')
#print(df.info())
#sns.countplot(x='benign_0__mal_1',data=df)
#df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')
#plt.show()
X=df.drop('benign_0__mal_1',axis=1).values
y=df['benign_0__mal_1'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

model= Sequential()


model.add(Dense(30,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid')) #Binary Classification
model.compile(loss='binary_crossentropy',optimizer='adam')
#model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),verbose=3)

from tensorflow.keras.callbacks import EarlyStopping  
early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)
model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),verbose=3,callbacks=[early_stop])

loss_df=pd.DataFrame(model.history.history)
loss_df.plot()
plt.show()

predictions = (model.predict(X_test) > 0.5).astype("int32")
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))








