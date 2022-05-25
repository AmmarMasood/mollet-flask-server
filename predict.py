from flask import Flask,request,jsonify
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def predict_sentiment(text, model_dict):
  
    # pipe = Pipeline([
    #     ('vect', model_dict['vectorizer']),  # Transform text into tokens using individual words, pairs and triplets
    #     ('tfidf', model_dict['tfidf']),  # Take into account word frequency
    #     ('clf', model_dict['model'])  # Classification model passed to the function
    # ])
    # print("Model loaded", loaded_model)
    # make prediction
    # pipe = Pipeline([
    #       ('clf', model_dict['model']) 
    # ])
    # text = [text]
    prediction = model_dict.predict(text)
    return prediction;