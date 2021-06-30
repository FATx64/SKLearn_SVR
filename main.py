# SKLearn SVR 

import pandas as pd

df = pd.read_csv('Salary_Data.csv')

#mengecek data
df.info()

#menampilkan 5 baris dari data
df.head()

#pisahkan antara atribut dan label yang ingin diprediksi.
import numpy as np
 
# memisahkan atribut dan label
X = df['YearsExperience']
y = df['Salary']
 
# mengubah bentuk atribut
X = X[:,np.newaxis]

# buat objek support vector regression dan di sini kita akan mencoba 
# menggunakan parameter C = 1000, gamma = 0.05, dan kernel ‘rbf’

from sklearn.svm import SVR
 
# membangun model dengan parameter C, gamma, dan kernel
model  = SVR(C=1000, gamma=0.05, kernel='rbf')
 
# melatih model dengan fungsi fit
model.fit(X,y)

#visualisasikan bagaimana model SVR kita menyesuaikan terhadap pola yang terdapat pada 
#data menggunakan library matplotlib.

import matplotlib.pyplot as plt
 
# memvisualisasikan model
plt.scatter(X, y)
plt.plot(X, model.predict(X))
