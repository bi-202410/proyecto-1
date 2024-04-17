# import nltk
import re, string, unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

nltk.download('stopwords')
nltk.download('punkt')

import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

#Pipeline
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def tokenize_text(text):
    stop_words = set(stopwords.words('spanish'))
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"
                            u"\U0001F300-\U0001F5FF"
                            "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    stemmer = SnowballStemmer('spanish')
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def get_top_features(pipeline):
    # Obtener el vectorizador TF-IDF del pipeline
    tfidf_vectorizer = pipeline.named_steps['tfidf']
    # Obtener los nombres de las características
    feature_names = tfidf_vectorizer.get_feature_names_out()
    # Obtener el modelo del pipeline
    model = pipeline.named_steps['model']
    coefficients = model.coef_

    # Inicializar la lista de palabras más importantes por score
    top_words_by_score = []

    # Obtener las palabras más influyentes para cada score
    for i, score_coefficients in enumerate(coefficients):
        sorted_indices = np.argsort(score_coefficients)
        top_indices = sorted_indices[-10:]  # Obtener los índices de las 10 palabras principales
        top_indices = top_indices[::-1]  # Invertir para obtener las palabras con los coeficientes más altos primero
        top_words = [feature_names[idx] for idx in top_indices]
        top_words_by_score.append(top_words)

    return top_words_by_score

def train_evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test):
    # Entrenar el pipeline
    pipeline.fit(X_train, y_train)
    dump(pipeline, 'assets/pipeline.joblib')

    # Predecir en los conjuntos de entrenamiento y prueba
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Calcular métricas de evaluación para el conjunto de entrenamiento
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')

    # Calcular métricas de evaluación para el conjunto de prueba
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Devolver las métricas de evaluación
    return {
        'Train Accuracy': train_accuracy,
        'Train Recall (Weighted)': train_recall,
        'Train F1-Score (Weighted)': train_f1,
        'Test Accuracy': test_accuracy,
        'Test Recall (Weighted)': test_recall,
        'Test F1-Score (Weighted)': test_f1
    }

