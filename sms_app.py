from flask import Flask, request, jsonify,render_template
import joblib  # You can use your trained model here
import re
import os


# Load your trained model (adjust the file path as needed)
model = joblib.load('./NLP_Projects/SMS_Spam_Detection_NB_Model.pkl')
cv= joblib.load('./NLP_Projects/TF_DIF.pkl')

# Define the path to templates and static folder
templates_folder = os.path.join(os.path.dirname(__file__), 'templates')
static_folder = os.path.join(os.path.dirname(__file__), 'static')
print(f"templates_folder is {templates_folder}")
print(f"static_folder is {static_folder}")

app = Flask(__name__,template_folder=templates_folder,static_folder=static_folder)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        SMS = request.form['SMS']
        data = [SMS]
        vector = cv.transform(data).toarray()
        prediction = model.predict(vector)
        # Convert prediction to a human-readable format
        #result = 'spam' if prediction[0] == 1 else 'ham'
        # Return prediction back to the front-end (render template with prediction)
        return render_template('result.html', prediction=prediction)
if __name__ == '__main__':
    app.run(debug=True)
