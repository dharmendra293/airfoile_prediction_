import pickle
from flask import Flask, request, jsonify,app,url_for,render_template
from flask_cors import CORS
from flask import Response
import numpy as np
import pandas as pd
import joblib
app = Flask(__name__)
CORS(app)
# load the model
model = pickle.load(open('airfoil_noise_model.pkl', 'rb'))
scaler = joblib.load('scaler.pkl')
@app.route('/')
def home():
    return render_template("home.html") 

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json["data"]
    print(data)
    # convert data into numpy array
    da=np.array([data['Frequency'], data['Angle of attack'], data['Chord length'], data['Free-stream velocity'], data['Suction side displacement thickness']]).reshape(1, -1)
    # scale the data
    da = scaler.transform(da)
    # make prediction
    prediction = model.predict(da)
    # return the prediction
    return jsonify({'prediction': prediction[0]})

    


if __name__ == "__main__":
    app.run(debug=True)
