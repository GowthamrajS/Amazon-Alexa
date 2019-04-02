"""
   About the Data

    This dataset consists of a nearly 3000 Amazon customer reviews (input text), star ratings, date of review, variant and
    feedback of various amazon Alexa products like Alexa Echo, Echo dots, Alexa Firesticks etc. for learning how to train Machine for sentiment analysis.

   Source
   
   Extracted from Amazon's website

        You can use this data to analyze Amazon’s Alexa product ; discover insights into consumer reviews and
        assist with machine learning models.
        
        You can also train your machine models for sentiment analysis and
        analyze customer reviews how many positive reviews ?
        and how many negative reviews ?
        
   
   

"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sea
import missingno
# Importing the dataset

dataset = pd.read_csv('amazon_alexa.tsv', sep = '\t')

y = dataset["feedback"].values


missingno.matrix(dataset,figsize = (6,5))

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

for i in range(0, 3150):
    review = re.sub('[^a-zA-Z]', ' ', str(dataset['verified_reviews'][i]))
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    dataset['verified_reviews'][i] = review





# Creating Dummies 

dummie  =pd.get_dummies(dataset.variation,drop_first=True)

x = dataset.drop(["variation","date","feedback"],axis =1)

train  = pd.concat([x,dummie],axis=1)



# visualization 


plt.figure(figsize = (20,10))
sea.barplot(data = dataset , y = "rating",x = "variation")





from sklearn.feature_extraction.text import CountVectorizer as cv

cv =cv()

cv = cv.fit_transform(train["verified_reviews"])

cv = pd.DataFrame(cv.toarray())

train = pd.concat([train,cv],axis =1)

train = train.drop(["verified_reviews"],axis =1)

train.shape


# train , test and split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(train,y, test_size = 0.2)


# Using Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier as df

df = df()

df.fit(x_train,y_train)

df.score(x_train,y_train)


# Predicted in train set 

x_pred = df.predict(x_train)


# Calculating X_train data how much Accuracy fitted in the model

from sklearn.metrics import classification_report, confusion_matrix
cr =  classification_report (y_train,x_pred)
cm =    confusion_matrix(y_train,x_pred)


sea.heatmap(cm,annot =True,fmt="g",cbar = False)



# Predicted in test data

y_pred = df.predict(x_test)


#Calculating X_train data how much Accuracy fitted in the model

from sklearn.metrics import classification_report, confusion_matrix
cr =  classification_report (y_test,y_pred)
cm =    confusion_matrix(y_test,y_pred)

sea.heatmap(cm,annot =True,fmt="g",cbar = False)






"""
     100% Accuracy in DecisionTreeClassifier
     
Kaggle:
    
     https://www.kaggle.com/sid321axn/amazon-alexa-reviews

"""













