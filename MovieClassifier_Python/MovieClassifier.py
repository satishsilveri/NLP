# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import re
from sklearn.datasets import load_files
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class TextProcessor:
    
    def pre_process_documents(self,input_data):
        
        documents=[]
        
        stemmer = WordNetLemmatizer()
        
        for sentence in range(0,len(X)):
            
            #remove special character
            document = re.sub(r'\W',' ',str(X[sentence]))
            
            #remove single character
            document = re.sub(r'\s+[a-zA-z]\s+',' ',document)
            
            #convert double space to single space
            document = re.sub(r'\s+',' ',document)
            
            #remove b prefix
            document = re.sub(r'^b\s+', '', document)
            
            #to lower case
            document = document.lower()
            
            #lemmetization
            #words = nltk.word_tokenize(document)
            
            words= document.split()
            
            document = [stemmer.lemmatize(word) for word in words]
            
            document = ' '.join(document)
            
            documents.append(document)
            
        return documents
                 
    def words_to_vectors(self,cleansed_data):
        
        tfidf_converter= TfidfVectorizer(max_features=1500,min_df=5,max_df=0.7,stop_words=stopwords.words('english'))
        
        return tfidf_converter.fit_transform(cleansed_data).toarray()   
        
    

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
    
    with open('movie_review_classifier', 'wb') as model:  
        pickle.dump(classifier,model)
        
    print("Model saved successfully...")


