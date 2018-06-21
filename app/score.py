from flask import Flask  
from flask import request  
import pickle  
import numpy as np

categories = ["Setosa", "Versicolor", "Virginica"]

# flask is the web server
app = Flask(__name__)  

#loading the model
print("Import the model from model.pkl ")
file = open('model.pkl', 'rb')
model = pickle.load(file)

def predict(x):
    if x is None:
        x = [[3.0, 3.6, 1.3, 0.25]]
    print ('New sample: {}'.format(x))
    pred = model.predict(x)
    print('Predicted class is '+ categories[pred[0]])
    return categories[pred[0]]

@app.route('/')  
def default():
    return 'default'

    
@app.route('/version')  
def version():
    return '1.3'

@app.route('/api/predict', methods=['POST'])  
def run():
    data = request.json
    predicted = predict([data])
    return predicted

if __name__ == "__main__":
    predict(None)#test run 
    app.run(port=7777,host='0.0.0.0')  
