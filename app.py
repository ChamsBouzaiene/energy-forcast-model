from flask import Flask, request
from keras.models import load_model
import pandas as pd
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the model
model = load_model("static/myModel.h5", compile=False)
graph = tf.get_default_graph()

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict')
def predict():
    # global graph
    inputString = request.args['inputdata']
    inputData = np.array([float(i) for i in inputString.split(',')])
    inputData = inputData.reshape(1, 1, 11)
    with graph.as_default():
        prediction = model.predict(inputData)
    return str(prediction[0][0])


if __name__ == '__main__':
    app.run()
