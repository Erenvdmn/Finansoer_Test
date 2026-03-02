import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from feature_engineering import FeatureEngineer


class DataPreparation:
    def __init__(self, filepath, seq_length=30):
        self.filepath = filepath
        self.seq_length = seq_length,
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
    

if __name__ == "__main__":
    dp = DataPreparation("data/AAPL_daily.csv", seq_length=30)
    X_train, X_test, y_train, y_test = dp.get_data()

    print("Data preparetion is complated! Pytorch Matris Sizes:")
    print(f"X_traing shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")