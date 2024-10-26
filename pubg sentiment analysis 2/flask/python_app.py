from flask import Flask, render_template, request
import torch
import torch.nn as nn
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model_path = 'D:/pubg sentiment analysis/flask/lstm_sentiment_analysis_model.pth'
vectorizer_path = 'D:/pubg sentiment analysis/flask/vectorizer.pickle'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size = 128
num_layers = 2
output_size = 1
vectorizer = None
input_size = None

def load_model():
    global model_text, vectorizer, input_size
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
            input_size = len(vectorizer.get_feature_names())
    except Exception as e:
        print(f"Error loading vectorizer: {e}")
        vectorizer = None
        input_size = None

    if input_size is not None:
        model_text = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
        model_text.load_state_dict(torch.load(model_path, map_location=device))
        model_text.eval()

load_model()

def preprocess_text(text):
    return text

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/index.html')
def predict():
    return render_template("index.html")

@app.route('/output.html', methods=["GET", "POST"])
def output():
    result_message = None
    if request.method == "POST":
        text = request.form["text"].strip()
        if text == "":
            result_message = "Error: Empty input."
        else:
            text_preprocessed = preprocess_text(text)
            if vectorizer is None or model_text is None:
                result_message = "Error: Model or vectorizer not loaded."
            else:
                text_vectorized = vectorizer.transform([text_preprocessed])
                with torch.no_grad():
                    outputs = model_text(torch.tensor(text_vectorized.toarray(), dtype=torch.float32).to(device).unsqueeze(1))
                    prediction = torch.sigmoid(outputs.squeeze()).item()
                    if prediction >= 0.5:
                        result_message = f"Positive sentiment (Confidence: {prediction:.2f})"
                    else:
                        result_message = f"Negative sentiment (Confidence: {1 - prediction:.2f})"
    return render_template("output.html", result=result_message)

if __name__ == "__main__":
    app.run(debug=True)
