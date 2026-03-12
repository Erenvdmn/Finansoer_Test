# Finansoer: AI-Powered Financial Risk Analyzer

Finansoer is a comprehensive, fully automated financial analysis pipeline. It is designed to evaluate stock market risks by merging quantitative machine learning metrics with qualitative real-time news analysis driven by Generative AI. 

Instead of relying solely on numbers or solely on news, Finansoer looks at both. It calculates the statistical probability of a stock crashing and then asks an AI to verify if the real-world news supports that data.

## 🧠 How the Pipeline Works

The entire system is orchestrated through a single script (`main.py`) and operates in three major automated steps:

### Step 1: Automated Data Collection & Feature Engineering
- **Data Pulling:** The system dynamically fetches the maximum available historical daily data for a specified stock ticker using the `yfinance` API. It also handles local caching, so it only downloads missing days on subsequent runs.
- **Math & Indicators:** It automatically calculates critical technical analysis indicators, including the 20-Day and 50-Day Simple Moving Averages (SMA), 14-Day Relative Strength Index (RSI), MACD, and 20-Day Volatility.
- **Target Labeling:** It looks 5 trading days into the future to create a binary target label: *Will the stock price drop by more than 5%?*

### Step 2: Machine Learning Risk Scoring (XGBoost & SHAP)
- **Model Training:** A custom XGBoost classifier is trained on the historical data and calculated indicators.
- **Probability Calibration:** The model's outputs are calibrated to provide a realistic probability percentage (Risk Score) of a sharp decline occurring in the next 5 days.
- **Explainability:** Using SHAP (SHapley Additive exPlanations), the system breaks down exactly which indicators increased or decreased the risk score and saves a visual plot (`shap_explanation.png`).

### Step 3: Generative AI Contextual Analysis (RAG Pipeline)
- **News Scraping:** The system scrapes the company's business summary and the 5 most recent news headlines.
- **Vector Database:** It chunks the text and builds a local vector database using FAISS and HuggingFace embeddings (`all-MiniLM-L6-v2`).
- **LLM Reasoning:** A locally hosted Llama 3 model (via Ollama and LangChain) acts as a Senior Quant Analyst. It reads the numeric risk score from Step 2 and cross-references it with the real-time news to determine if there are hidden fundamental risks or mitigating positive catalysts.

## 💻 Tech Stack
- **Core:** Python, Pandas, NumPy
- **Machine Learning:** XGBoost, Scikit-Learn, SHAP
- **Generative AI & RAG:** LangChain, FAISS, HuggingFace, Ollama (Llama 3)
- **Financial Data:** yfinance

## 🚀 Usage

1. Open `main.py`.
2. Change the `POINTED_TICKER` variable to the stock symbol you want to analyze (e.g., `"NVDA"`, `"TSLA"`, `"MSFT"`).
3. Run the pipeline from your terminal:

   python main.py

The system will print out the step-by-step process, the final numerical risk score, the SHAP breakdown, and the Llama 3 analyst report.

## 🗺️ Future Roadmap
- **Streamlit Web Dashboard:** Migrating the terminal output to an interactive web interface with live charts and UI buttons.
- **Deep Learning Integration:** Adding an LSTM (Long Short-Term Memory) neural network alongside XGBoost for more complex time-series pattern recognition.
- **Portfolio Analysis:** Expanding the capability to analyze multiple stocks simultaneously for portfolio risk balancing.