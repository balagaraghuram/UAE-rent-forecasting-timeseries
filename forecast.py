import pandas as pd
import torch
from tensorflow.keras.models import load_model
import joblib

def forecast_pytorch(data, seq_len=12):
    model = LSTMModel(input_dim=1, hidden_dim=64, num_layers=2)
    model.load_state_dict(torch.load('models/pytorch_model.pt'))
    model.eval()

    input_seq = data[-seq_len:]
    input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        prediction = model(input_seq).item()
    return prediction

def forecast_tf(data, seq_len=12):
    model = load_model('models/tensorflow_model.h5')
    input_seq = np.array(data[-seq_len:]).reshape(1, seq_len, 1)
    prediction = model.predict(input_seq)
    return prediction[0][0]

def forecast_rf(data, seq_len=12):
    model = joblib.load('models/random_forest_model.pkl')
    input_seq = data[-seq_len:]
    prediction = model.predict([input_seq])
    return prediction[0]
