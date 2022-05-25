from flask import Flask,request,jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from predict import predict_sentiment
import json



app = Flask(__name__, template_folder="template")
cors = CORS(app, resources={r".*": {"origins": "*"}})

UPLOAD_FOLDER = '/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# # load the model from disk
# loaded_model = pickle.load(open('model/sentiment.pkl', 'rb'))
# print("Model loaded", loaded_model)

loaded_model = pickle.load(open('model/senti.pkl', 'rb'))
# vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
# tfidf = pickle.load(open('model/tfidf.pkl', 'rb'))

# model_dict = {"model": loaded_model, 'vectorizer':vectorizer, 'tfidf':tfidf}
@app.route('/predict', methods=['POST'])
def predict_sentiment_1():
    
    # load the model from disk
    text = request.get_json()['text']
    sentiment = predict_sentiment(text, loaded_model)
    listToStr = ' '.join([str(elem) for elem in sentiment])
    # print(type(sentiment), sentiment)
    # listToStr = json.dumps([sentiment])
    return listToStr
    


if __name__ == "__main__":
    app.run(debug=True)