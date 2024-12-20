from flask import Flask, request, jsonify
import joblib

rf_model = joblib.load('random_forest_model.pkl')

# Initialize flask App
app = Flask(__name__)


# Define the prediction end point
@app.route('/',methods=['GET'])
def home():
    return "Welcome to the Random Forest Prediction API!"

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()

    input_data = data['input']

    prediction = rf_model.predict([input_data])

    return jsonify({'prediction':prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)


'''
For Invoking Predict Endpoint use below curl command.

curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"input": [1.2, 3.4, 5.6, 7.8]}' 

'''
# Input should be embedded values with 100 dimension.
    