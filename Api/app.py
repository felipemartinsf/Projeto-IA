from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sklearn
import re
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Configurações do Flask
app = Flask(__name__)
CORS(app)

@app.route('/classify', methods=['POST'])
def classify_news():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'Texto não fornecido'}), 400
    
    # Carregar o modelo e tokenizer
    tokenizer = BertTokenizer.from_pretrained('C:\\Users\\pedro\\OneDrive\\Desktop\\Ia2\\Projeto-IA\\codes\\token')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(torch.device('cpu'))

    # Carregar os parâmetros salvos
    model.load_state_dict(torch.load('melhormodelobert.pkl', map_location=torch.device('cpu')))

    # Configurar o modelo para avaliação
    model.eval()

    # Fazer previsões
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=256) 
    inputs = {key: val.to(torch.device('cpu')) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        prediction = torch.argmax(outputs.logits, dim=1).item()

    # Interpretação do resultado
    if prediction == 1:
        print("Fake news")
    else:
        print("News")

    return jsonify({'isFake': bool(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
