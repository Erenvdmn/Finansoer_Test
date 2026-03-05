import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch 

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from feature_engineering import FeatureEngineer

import torch.nn as nn
import torch.optim as optim


class DataPreparation:
    def __init__(self, filepath, seq_length=30):
        self.filepath = filepath
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(0,1))


    def get_data(self):
        print("Datas pulling from FeatureEngineer...")
        engineer = FeatureEngineer(self.filepath)
        df = engineer.get_processed_data()

        features = ['Close', 'Daily_Return', 'Volatility', 'RSI_14', 'MACD', 'SMA_20', 'SMA_50']
        target = 'Target_Downside'

        scaled_features = self.scaler.fit_transform(df[features])
        targets = df[target].values

        print(f"creating {self.seq_length} days time windows...")
        # Last 30 days = X, point = y
        X, y = [], []
        for i in range((self.seq_length), len(scaled_features)):
            X.append(scaled_features[i - self.seq_length : i])
            y.append(targets[i])

        X, y = np.array(X), np.array(y)


        # %80 Learning, %20 Test
        split_idx = int(len(X)*0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]


        print("Datas are being transform to PyTroch Tensors...")
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        return X_train, X_test, y_train, y_test
    



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


def train_model(X_train, y_train):
    input_size = 7
    hidden_size = 128
    num_layers = 2
    output_size = 1
    num_epochs = 100
    learning_rate = 0.0005

    model = DownsideLSTM(input_size, hidden_size, num_layers, output_size)
    critertion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Epoch starting...")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = critertion(outputs, y_train)

        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model


if __name__ == "__main__":
    dp = DataPreparation("data/AAPL_daily.csv", seq_length=30)
    X_train, X_test, y_train, y_test = dp.get_data()

    trained_model = train_model(X_train, y_train)

    torch.save(trained_model.state_dict(), "models/downside_lstm.pth")
    print("\n Model sucsessfully trained and saved as models/downside_lstm.pth")