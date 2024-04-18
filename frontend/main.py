import requests
import os

from flask import Flask, render_template, request
from dotenv import load_dotenv
import json
import pandas as pd

load_dotenv()
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    endpoint = f'{BACKEND_URL}/predict'
    response = requests.post(endpoint, json=[{'review': review}])
    prediction = response.json()
    review = prediction['review']
    pred_num = prediction['prediction'][0]
    pred_num = [round(num * 100, 0) for num in pred_num]
    num_classes = len(pred_num)
    predicted_class = pred_num.index(max(pred_num)) + 1
    classes = ['Muy Negativa', 'Negativa', 'Neutra', 'Positiva', 'Muy Positiva']
    return render_template('index.html',
                           prediction=pred_num,
                           review=review,
                           num_classes=num_classes,
                           predicted_class = predicted_class,
                           classes = classes)

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/retrain', methods=['POST'])
def retrain():
    endpoint = f'{BACKEND_URL}/retrain'

    if 'train_data' not in request.files:
        return 'No file part'
    file = request.files['train_data']
    if file.filename == '':
        return 'No selected file'
    if not allowed_file(file.filename):
        return 'Invalid file type'

    file_df = pd.read_csv(file, encoding='latin1')
    file_df.rename(columns={'Class': 'class_', 'Review': 'review'}, inplace=True)
    payload = file_df.to_dict(orient='records')
    print(payload)
    response = requests.post(endpoint, json=payload)

    # Parse the prediction from the API response
    model_results = response.json()

    print(model_results)

    # Render the HTML template with the prediction
    return render_template('train.html', model_results=str(model_results))

if __name__ == '__main__':
    app.run(debug=True)