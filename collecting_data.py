import os
import pandas as pd
import yfinance as yf
from datetime import datetime

class DataDownloader:
    def __init__(self, data_folder='data'):
        self.data_folder = data_folder

        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            print(f"📁 Created data folder at: {self.data_folder}")

    def get_daily_data(self, ticker):
        # Fetches daily stock data from local cache or yfinance.
        file_path = os.path.join(self.data_folder, f"{ticker}_daily.csv")
        
        # 1. IF FILE EXISTS IN LOCAL STORAGE
        if os.path.exists(file_path):
            print(f"✅ Loading {ticker} data from local disk...")
            
            df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
            
            # Make sure the index is timezone-naive so comparisons work
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            last_date = df.index.max()
            today = pd.Timestamp.today().normalize()
            
            # Check if the local data is up-to-date (ignoring weekends)
            if last_date >= today - pd.Timedelta(days=1):
                print("🌟 Data is up-to-date!")
                return df
            else:
                print(f"🔄 Data is outdated (Last date: {last_date.date()}). Updating...")
                try:
                    # Download ONLY the missing days! (This saves us from Rate Limits)
                    start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    new_data = yf.download(ticker, start=start_date, progress=False)

                    if not new_data.empty:
                        # Clean timezone if exists
                        if new_data.index.tz is not None:
                            new_data.index = new_data.index.tz_localize(None)
                            
                        # Merge old and new data
                        combined_df = pd.concat([df, new_data])
                        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                        combined_df.sort_index(inplace=True)

                        # Save updated dataframe
                        combined_df.to_csv(file_path)
                        print(f"✅ {ticker} data successfully updated!")
                        return combined_df
                    else:
                        print("⚠️ No new trading days found, returning existing data.")
                        return df
                        
                except Exception as e:
                    print(f"❌ Error while updating: {e}")
                    print("⚠️ Using existing old data as fallback.")
                    return df

        # 2. IF FILE DOES NOT EXIST (First time download)
        print(f"🚀 Downloading full historical data for {ticker} using yfinance...")
        try:
            # yf.download is much more stable than yf.Ticker().history()
            data = yf.download(ticker, period="max", progress=False)

            if not data.empty:
                # Clean timezone for consistency
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                    
                # Save the full historical data
                data.to_csv(file_path)
                print(f"💾 {ticker} data saved successfully to '{file_path}'")
                return data
            else:
                print(f"❌ Could not download data for {ticker}. It might be delisted or invalid.")
                return None
            
        except Exception as e:
            print(f"❌ Error occurred while downloading full data: {e}")
            return None

# for manuel testing of this file
if __name__ == "__main__":
    downloader = DataDownloader()
    df = downloader.get_daily_data("AAPL")
    if df is not None:
        print(df.tail())