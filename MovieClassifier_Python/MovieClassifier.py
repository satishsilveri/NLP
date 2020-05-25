# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from sklearn.datasets import load_files
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from TextPreprocessor import TextProcessor


if __name__=="__main__":
    
    path= "path\to\txt_sentoken"
    
    text_processor=TextProcessor()
    
    
    #load data
    
    movie_data=load_files(path)
    
    X,y = movie_data.data,movie_data.target
    
    #clean the data
    
    X = text_processor.pre_process_documents(X)
    
    print("Text preprocessing done...")
    
    #convert words to vectors
    
    X = text_processor.words_to_vectors(X)
    
    print("converted words to vectors")
    
    #split the data into train and test sets
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    
    print("splitted data into train and test sets")
    
    #train model using RandomForest classifier
    
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    
    classifier.fit(X_train,y_train)
    
    print("trained classifier")
    
    y_pred=classifier.predict(X_test)


    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))  
    print(accuracy_score(y_test, y_pred)) 
    
    
    #save model
    
    with open('E:\Personal\ML_Journey\model\movie_classifier,pkl', 'wb') as model:  
        pickle.dump(classifier,model)
        
    print("Model saved successfully...")


