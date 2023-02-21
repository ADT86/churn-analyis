# churn-analyis
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt
%matplotlib inline

churn_df=pd.read_csv("AIA_Churn_Modelling_Case_Study.csv")
dataset = churn_df
dataset =  dataset.drop(["Dependents","Partner","customerID","PaymentMethod"], axis=1)
dataset=pd.get_dummies(data=dataset, columns=['PhoneService',"MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","TotalCharges","TotalCharges","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","gender"])
dataset['Churn'] = churn_df['Churn'].replace({"Yes":1,"No":0})
X =  dataset.drop(['Churn'],axis=1)
y = dataset['Churn']

#creating the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=200, random_state=0) 
classifier.fit(X_train, y_train) 
predictions = classifier.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,predictions )) 
print(accuracy_score(y_test, predictions ))

feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
