# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:43:15 2020

@author: satis
"""
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer



class TextProcessor:
    
    def pre_process_documents(self,input_data):
        
        documents=[]
        
        lemmatizer = WordNetLemmatizer()
        
        for sentence in range(0,len(input_data)):
            
            #remove special character
            document = re.sub(r'\W',' ',str(input_data[sentence]))
            
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
            
            document = [lemmatizer.lemmatize(word) for word in words]
            
            document = ' '.join(document)
            
            documents.append(document)
            
        return documents
                 
    def words_to_vectors(self,cleansed_data):
        
        tfidf_converter= TfidfVectorizer(max_features=1500,min_df=5,max_df=0.7,stop_words=stopwords.words('english'))
        
        return tfidf_converter.fit_transform(cleansed_data).toarray()