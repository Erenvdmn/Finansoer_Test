import pandas as pd

class FeatureEngineer:
    def __init__(self, file_path):
        print(f"Loading data from {file_path}")

        # Reading and cleaning the multi-index columns
        self.df = pd.read_csv(file_path, header=[0,1], index_col=0)
        self.df.columns = self.df.columns.droplevel(1)
        self.df.columns.name = None

        self.df = self.df[['Close', 'High', 'Low', 'Open', 'Volume']]

    # Calculates Simple Moving Averages (SMA)
    def add_moving_averages(self):
        print("Calculating Moving Averages (SMA_20, SMA_50)...")
        # Rolling window takes the last X days and calculates the mean
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()

    # Calculates daily percentage returns
    def add_daily_returns(self):
        print("Calculating Daily Returns...")
        self.df['Daily_Return'] = self.df['Close'].pct_change()


    # Calculates Relative Strength Index (RSI) using Wilder's Smoothing (EWM)
    def add_rsi(self, window=14):
        print(f"Calculating RSI ({window} days) with Exponential Moving Average...")

        delta = self.df['Close'].diff()

        gains = delta.copy()
        loss = delta.copy()
        gains[gains < 0] = 0
        loss[loss > 0] = 0

        avg_gain = gains.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = abs(loss.ewm(com=window - 1, min_periods=window).mean())

        rs = avg_gain / avg_loss
        self.df['RSI_14'] = 100 - (100 / (1 + rs))

    
    # Calculates 20-day Rolling Volatility (Risk Measure).
    def add_volatility(self, window=20):
        print(f"Calculating Volatility ({window} days)...")

        # Statistical deviation of daily returns (Risk)
        self.df['Volatility'] = self.df['Daily_Return'].rolling(window=window).std()

    
    # Calculates MACD and Signal Line
    def add_macd(self):
        print("Calculating MACD...")
        # Daily Exponantial Averages (EMA) of 12 and 26 days
        ema_12 = self.df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = self.df['Close'].ewm(span=26, adjust=False).mean()

        # MACD Line and 9 Days Signal Line
        self.df['MACD'] = ema_12 - ema_26
        self.df['MAC_Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()


    # Creates the target variable for Machine Learning.
    # Looks ahead 'look_forward' days. If the price drops by more than 'threshold' (e.g., 5%), label is 1. Otherwise 0.
    def add_target_label(self, look_forward=5, threshold=-0.05):
        print(f"🎯 Creating Target Label (Looking {look_forward} days ahead for a {threshold*100}% drop)...")
        
        future_returns = self.df['Close'].pct_change(periods=look_forward).shift(-look_forward)
        
        self.df['Target_Downside'] = (future_returns <= threshold).astype(int)

        self.df.loc[future_returns.isna(), 'Target_Downside'] = None


    # Runs all feature methods and return the cleaned DataFrame
    def get_processed_data(self):
        self.add_moving_averages()
        self.add_rsi()
        self.add_daily_returns()
        self.add_volatility()
        self.add_macd()

        self.add_target_label(look_forward=5, threshold=-0.05)

        self.df.dropna(subset=['SMA_50', 'MACD', 'RSI_14'], inplace=True)
        return self.df
    
        
# TESTING
if __name__ == "__main__":
    engineer = FeatureEngineer("data/AAPL_daily.csv")
    processed_df = engineer.get_processed_data()

    print("\n Feature Engineering Complete! Here is the latest data:")

    print(processed_df[['Close', 'SMA_20', 'SMA_50', 'Daily_Return', 'Volatility', 'RSI_14', 'MACD', 'Target_Downside']].tail())