# FinansOer: Hybrid AI Risk Analysis Engine 🚀

FinansOer is an enterprise-grade, fully automated financial risk analysis backend. It merges quantitative Machine Learning (Deep Learning & Tree-based models) with qualitative Document Intelligence (RAG & LLMs) to predict stock market crash risks and provide institutional-level explanations.

## 🏗️ System Architecture & Completed Milestones

We have successfully implemented the entire end-to-end pipeline:

- **1. Data Engineering (`collecting_data.py` & `feature_engineering.py`)**
  - Asynchronously pulls historical stock data via `yfinance`.
  - Dynamically calculates technical indicators (SMA-20, SMA-50, RSI-14, MACD, Volatility).
- **2. Hybrid Quantitative Engine (`model_training.py` & `risk_scoring.py`)**
  - **PyTorch LSTM:** A Deep Learning sequential model that captures time-series patterns.
  - **XGBoost & SHAP:** A calibrated tabular model that ingests LSTM predictions and standard indicators to output a probabilistic crash risk score. Explainable AI (SHAP) is used to map feature importances.
- **3. Institutional RAG Pipeline (`rag_pipeline.py`)**
  - Connects to the SEC EDGAR database via API to fetch the latest official **10-K filings** (Item 1A: Risk Factors).
  - Uses `FAISS` and `HuggingFace` embeddings (`all-MiniLM-L6-v2`) to vectorize corporate documents.
  - Prompts a locally hosted **Llama 3** (via Ollama) to synthesize the numerical XGBoost risk score with the actual corporate SEC texts.
- **4. Microservice & Containerization (`main.py` & `Dockerfile`)**
  - Fully wrapped in an asynchronous **FastAPI** application.
  - **Dockerized** for cross-platform, environment-agnostic deployment (handling local host LLM routing via `host.docker.internal`).

---

## ⚙️ Prerequisites

Before running the application, ensure you have the following installed on your host machine:
1. **Docker Desktop** (Make sure the Docker engine is running).
2. **Ollama** (Running locally with the Llama 3 model pulled: `ollama run llama3`).
3. An active **SEC API Key** (from sec-api.io).

---

## 🚀 How to Run (Docker)

**1. Set up your environment variables**
Create a `.env` file in the root directory and add your SEC API Key:
```env
SEC_API_KEY=your_api_key_here
```

**2. Build the Docker Image**
Package the application into an isolated container:
```bash
docker build -t finansoer .
```

**3. Run the Container**
Start the FastAPI server. (The application maps port 8000 and connects to your local Ollama instance):
```bash
docker run -p 8000:8000 finansoer
```

**4. Trigger the AI Engine**
Once the server is running, you can analyze any stock (e.g., TSLA, NVDA, AAPL) by visiting the auto-generated Swagger UI:
- **Docs:** `http://localhost:8000/docs`
- **Direct API Call:** `http://localhost:8000/analyze/TSLA`

The API will return a JSON response containing the numerical risk score and a highly professional, context-aware LLM synthesis report.
