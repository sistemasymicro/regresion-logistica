import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb

dataframe = pd.read_csv(r"bank.csv")
print(dataframe.head())

print(dataframe.describe())
print(dataframe.groupby('y').size())

dataframe.drop(['y'],1).hist()
plt.show()

sb.pairplot(dataframe.dropna(), hue='y',size=4,vars=["balance", "campaign","previous","duration"],kind='reg')

X = np.array(dataframe.drop(['y'],1))
y = np.array(dataframe['y'])
X.shape

model = linear_model.LogisticRegression()
model.fit(X,y)

predictions = model.predict(X)
print(+predictions[0:5])
print(model.score(X,y))



