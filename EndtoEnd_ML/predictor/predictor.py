# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:33:16 2020

@author: satis
"""
from flask import Flask, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import pickle

app = Flask(__name__)

vectorizer_path=r'D:\Personal\ML_Journey\model\vectorizer.pkl'
classifier_path=r'D:\Personal\ML_Journey\model\spam_classifier.pkl'


def pre_process_message(lemmatizer,message):
    
    # tokenize the message
    words = nltk.word_tokenize(message)
    
    # remove stopwords
    cleaned_words = [word for word in words if word not in set(stopwords.words('english'))]
    
    # remove special characters
    filtered_words = [word for word in cleaned_words if word not in set(punctuation)]
    
    # remove words with length<=3
    
    pruned_words = [word for word in filtered_words if len(words)>3]
    
    # lemmatize the tokens
    
    lemmas = [lemmatizer.lemmatize(word) for word in pruned_words]
    
    return ' '.join(lemmas)



@app.route('/predict', methods=['POST'])
def predict():
    
    message = request.json['message']
    
   #message = input_json['message']
    
    lemmatizer = WordNetLemmatizer()
    
    # process raw text
    processed_text = pre_process_message(lemmatizer,message)
    
    # load vectorizer model
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))
    
    # create features
    vectorized_msg = vectorizer.transform([processed_text])
    
    # load classifier
    classifier = pickle.load(open(classifier_path, 'rb'))
    
    # predict the message class
    prediction = classifier.predict(vectorized_msg)
    
    if prediction[0] == 0:
        
        return jsonify('{"class_label":"ham"}')
    
    elif prediction[0] == 1:
        
        return jsonify('{"class_label":"spam"}')



if __name__=="__main__":
    
    app.run(debug=False)
    
    