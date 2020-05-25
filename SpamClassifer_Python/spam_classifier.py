# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:46:41 2020

@author: satis
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from TextPreprocessor import TextProcessor


if __name__=="__main__":
    
    path= r"E:\Machine Learning\data\Spam_data.csv"
    
    text_processor=TextProcessor()
    
    
    #load data
    
    spam_data=pd.read_csv(path)
    
    X,y = spam_data['Message'],spam_data['Category']
    
    #clean the data
    
    X = text_processor.pre_process_documents(X)
    
    print("Text preprocessing done...")
    
    #convert words to vectors
    
    X = text_processor.words_to_vectors(X)
    
    print("converted words to vectors")
    
    #split the data into train and test sets
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    
    print("splitted data into train and test sets")
    
    #train model using RandomForest classifier
    print('Training the classifier')
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    
    classifier.fit(X_train,y_train)
    
    print("Trained the classifier")
    
    y_pred=classifier.predict(X_test)


    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))  
    print(accuracy_score(y_test, y_pred)) 
    
    
    #save model
    
    with open('E:\Personal\ML_Journey\model\movie_classifier.pkl', 'wb') as model:  
        pickle.dump(classifier,model)
        
    print("Model saved successfully...")


