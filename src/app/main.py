# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
#import pickle
#import joblib
import os

from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth

# Criação de uma app
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

# Habilitando autenticação na app
basic_auth = BasicAuth(app)

# Antes das APIs
colunas = ['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']

def load_model(file_name = 'xgboost_undersampling.json'):
    model_xgb_2 = xgb.Booster()
    model_xgb_2.load_model(file_name)
    return model_xgb_2

# Carregar modelo treinado
modelo = load_model('models/xgboost_undersampling.json')

# Rota de predição de scores
@app.route('/resultado/', methods=['POST'])
@basic_auth.required
def get_score():
    # Pegar o JSON da requisição
    dados = request.get_json()
    # Garantir a ordem das colunas
    payload = np.array([dados[col] for col in colunas])
    # Fazer predição
    payload = xgb.DMatrix([payload], feature_names=colunas)
    score = np.float64(modelo.predict(payload)[0])
    status = 'Aceito'
    if score <= 0.7:
        status = 'Não aceito'
    return jsonify(score=score, status=status)

# Nova rota - recebendo dados
@app.route('/resultado/<entry>')
@basic_auth.required
def show_cpf(entry):
    return 'Recebendo dados\nEntrada: %s'%entry

# Rota padrão
@app.route('/')
def home():
    return 'API de analise de seguro'

# Subir a API
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

