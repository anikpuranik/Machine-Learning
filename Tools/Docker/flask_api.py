# importing libraries
import pandas as pd
from flask import Flask, request
import pickle


# initializing flask
app = Flask(__name__)
pickle_in = open('docker.pkl', 'rb')
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Welcome All."


@app.route('/predict')
def predict_note_authentication():
    vari = request.args.get("variance")
    skew = request.args.get("skewness")
    curt = request.args.get("curtosis")
    entr = request.args.get("entropy")
    prediction=classifier.predict([[vari, skew,
                                    curt, entr
                                   ]])
    print(prediction)
    message = "The prediction value : " + str(prediction)
    return message


@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    dataset = pd.read_csv(request.files.get('file'))
    prediction = classifier.predict(dataset)
    message = "The prediction value : " + str(list(prediction))    
    return message


if __name__ == '__main__':
    app.run()
