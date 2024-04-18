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
        return render_template('train.html', error='No se ha seleccionado un archivo')
    file = request.files['train_data']
    if file.filename == '':
        return render_template('train.html', error='No se encontró el archivo.')
    if not allowed_file(file.filename):
        return render_template('train.html', error='El archivo no es un CSV.')

    file_df = pd.read_csv(file, encoding='latin1')

    if 'Class' not in file_df.columns or 'Review' not in file_df.columns:
        return render_template('train.html', error='El archivo no tiene las columnas requeridas, por favor, revise el formato.')

    file_df.rename(columns={'Class': 'class_', 'Review': 'review'}, inplace=True)
    payload = file_df.to_dict(orient='records')
    print(payload)
    
    response = requests.post(endpoint, json=payload)

    # Parse the prediction from the API response
    model_results = response.json()[0]
    test_accuray = model_results['stats']['test_accuray']
    test_recall = model_results['stats']['test_recall']

    stats = {
        'test_accuray': f"{test_accuray:.2%}",
        'test_recall': f"{test_recall:.2%}"
    }

    features = model_results['features']
    class_1 = features.get('0', [])
    class_2 = features.get('1', [])
    class_3 = features.get('2', [])
    class_4 = features.get('3', [])
    class_5 = features.get('4', [])

    features = {
        'class_0': '-'.join([f'"{item}"' for item in class_1]),
        'class_1': '-'.join([f'"{item}"' for item in class_2]),
        'class_2': '-'.join([f'"{item}"' for item in class_3]),
        'class_3': '-'.join([f'"{item}"' for item in class_4]),
        'class_4': '-'.join([f'"{item}"' for item in class_5])
    }

    # Render the HTML template with the prediction
    return render_template('train.html', model_results=features, stats=stats)

if __name__ == '__main__':
    app.run(debug=True)