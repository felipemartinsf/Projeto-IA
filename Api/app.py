from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sklearn
import re
'''import nltk
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

nltk.download('rslp')
nltk.download('punkt')
nltk.download('stopwords')

# Carrega o modelo treinado
with open('melhormodelosvm.pkl', 'rb') as file:
    modelo = pickle.load(file)

app = Flask(__name__)
CORS(app)

# Funções auxiliares para processamento de texto
def remove_urls(text):
    return re.sub(r'http\S+', '', text)

def uppercase_percentage(text):
    total_letters = sum(1 for char in text if char.isalpha())
    if total_letters == 0:
        return 0
    uppercase_count = sum(1 for char in text if char.isupper())
    percentage = (uppercase_count / total_letters)
    return percentage

def count_special_chars(text):
    special_chars = ['!']
    return sum(1 for char in text if char in special_chars)

def remove_special_characters(text):
    pattern = r'[^\w\s]'
    clean_text = re.sub(pattern, ' ', text)
    return clean_text

def remove_numbers(text):
    return re.sub(r'\b\w*\d\w*\b', '', text).strip()

def preprocessar_texto(texto):
    stemizador = RSLPStemmer()
    stop_words = set(stopwords.words('portuguese'))

    palavras = word_tokenize(texto.lower())

    palavras_processadas = [
        stemizador.stem(palavra) for palavra in palavras
        if palavra.isalnum() and palavra not in stop_words
    ]

    return palavras_processadas

fakeNewsList = ['lul', 'ment', 'faz', 'pod', 'tud', 'vai', 'terr', 'plan', 'brasil', 'bolsonar', 'tod', 'mai', 'mora']

@app.route('/classify', methods=['POST'])
def classify_news():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'Texto não fornecido'}), 400
    
    # Processo de pré-processamento
    text = remove_urls(text)
    text = remove_special_characters(text)
    text = remove_numbers(text)

    # Adiciona o texto ao DataFrame
    data_df = pd.DataFrame({
        'TEXT': [text],
        'RESULTADO': [None]
    })

    data_df['uppercase_percentage'] = data_df['TEXT'].apply(uppercase_percentage)
    data_df['special_char_count'] = data_df['TEXT'].apply(count_special_chars)

    palavras_processadas = preprocessar_texto(text)
    contagem = sum(palavra in fakeNewsList for palavra in palavras_processadas)
    data_df['contagem_palavras_filtradas'] = contagem

    # Preparar os dados para o modelo
    X = data_df.drop(columns=['RESULTADO','TEXT'])

    # Fazer previsão usando o modelo carregado
    print(data_df)
    previsao = modelo.predict(X)
    print(previsao[0])
    # Adiciona o resultado ao DataFrame
    data_df['RESULTADO'] = previsao
    print(data_df['RESULTADO'])
    # Retorna o resultado da previsão'''
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer

    # Carregar o modelo e tokenizer
    tokenizer = BertTokenizer.from_pretrained('/content/drive/MyDrive/IA-projeto/aplicação')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(torch.device('cpu'))

    # Carregar os parâmetros salvos
    model.load_state_dict(torch.load('/content/drive/MyDrive/IA-projeto/aplicação/melhor_modelo_bert.pkl', map_location=torch.device('cpu')))

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
