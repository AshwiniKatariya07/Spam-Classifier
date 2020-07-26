from flask import Flask, render_template, request, jsonify
from sklearn.externals import joblib
import nltk
import requests
import pickle
import numpy as np
import sklearn

app = Flask(__name__)
classifier = joblib.load("countervectors.pkl")

@app.route('/', methods=['GET'])
def Home():
    return  render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.form['text']
        email = [email]
        #message = [email]
       # mail = cv.tranform(message).toarray()
        prediction = classifier.predict(email)
        if prediction == 1:
            return render_template('index.html',prediction_text = 'oops! This email is Spam')
        else:
            return render_template('index.html',prediction_text ='Great! This email is not spam' )

if __name__ == '__main__':
    app.run(debug=True)

