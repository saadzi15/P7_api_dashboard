#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from flask import Flask, jsonify, request
import joblib
from lightgbm import LGBMClassifier

app = Flask(__name__)

PATH = r'C:\Users\nha5600\Desktop\BUREAU\FORMATION\Suite\P7\Predictions\\'

df = pd.read_csv(PATH+'test_df_2.csv')
print('df shape = ', df.shape)

#Chargement du modèle
load_clf = joblib.load(PATH+r"trained_model_sample_.joblib")

@app.route('/')
def index():
    return 'Welcome to my Flask API!'

@app.route('/credit/<id_client>')
def credit(id_client):
    print('id client = ', id_client)

    # Récupération des données du client en question
    ID = int(id_client)
    X = df[df['SK_ID_CURR'] == ID]

    #Isolement des features non utilisées
    ignore_features = ['Unnamed: 0', 'SK_ID_CURR', 'INDEX', 'TARGET']
    relevant_features = [col for col in df.columns if col not in ignore_features]
    X = X[relevant_features]
    print('X shape = ', X.shape)

    #Prédiction
    proba = load_clf.predict_proba(X)
    prediction = load_clf.predict(X)
    pred_proba = {
        'prediction': int(prediction),
        'proba': float(proba[0][0])
    }

    print('Nouvelle Prédiction : \n', pred_proba)

    return jsonify(pred_proba)


# lancement de l'application
if __name__ == '__main__':
    app.run(debug=True)