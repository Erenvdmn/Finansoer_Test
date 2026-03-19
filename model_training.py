import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch 

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim

class DownsideLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DownsideLSTM, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Deciding Layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class LSTM_Pipeline:
    def __init__(self, filepath, seq_length=30):
        self.filepath = filepath
        self.seq_length = seq_length
        self.features = ['Close', 'Daily_Return', 'Volatility', 'RSI_14','MACD','SMA_20','SMA_50']
        self.scaler = MinMaxScaler(feature_range=(0,1))

        filename = os.path.basename(self.filepath).split('_')[0]
        self.model_path = f'models/{filename}_lstm.pth'

        if not os.path.exists("models"):
            os.makedirs("models")

    def process_sequences(self, df):
        scaled_features = self.scaler.fit_transform(df[self.features])
        X = []
        for i in range(self.seq_length, len(scaled_features)):
            X.append(scaled_features[i - self.seq_length : i])
        return np.array(X)
    
    def train_model(self, df):
        print(f"Training Deep Learning (LSTM) for {self.filepath}...")

        X_data = self.process_sequences(df)
        y_data = df['Target_Downside'].values[self.seq_length:]

        valid_idx = ~np.isnan(y_data)
        X_train = torch.tensor(X_data[valid_idx], dtype=torch.float32)
        y_train = torch.tensor(y_data[valid_idx], dtype=torch.float32).unsqueeze(1)

        model = DownsideLSTM(input_size=len(self.features), hidden_size=128, num_layers=2, output_size=1)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)

        model.train()
        num_epochs=100
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"LSTM Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), self.model_path)
        print(f"LSTM Model saved to {self.model_path}")

    def add_lstm_predictions(self, df):
        # Is there a trained model for this ticker, if not then train_model(df)
        if not os.path.exists(self.model_path):
            self.train_model(df)
        else:
            print(f"Existing LSTM Model found at {self.model_path}, loading weigths...")

        model = DownsideLSTM(input_size=len(self.features), hidden_size=128, num_layers=2, output_size=1)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        X_data = self.process_sequences(df)
        X_tensor = torch.tensor(X_data, dtype=torch.float32)

        with torch.no_grad():
            predictions = model(X_tensor).numpy().flatten()

        # can't guess because first 30 days is NaN
        padded_preds = np.concatenate([np.full(self.seq_length, np.nan), predictions])

        df['LSTM_Risk'] = padded_preds
        return df