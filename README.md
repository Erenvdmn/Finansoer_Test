# FinansOer: AI-Powered Financial Risk Analyzer

## 📌 Overview

The Finansoer is a comprehensive financial data analysis and risk scoring pipeline. It dynamically processes financial data, performs feature engineering, trains machine learning models (XGBoost & LSTM), and provides LLM-backed qualitative risk assessments. The project is fully containerized using Docker and served as a RESTful API via FastAPI.

---

## 🏗️ Project Structure

```text
collecting_data.py       # Fetches and preprocesses stock data via yfinance
feature_engineering.py   # Calculates technical indicators (SMA, RSI, MACD, etc.)
risk_scoring.py          # Calculates numerical risk using XGBoost & SHAP
model_training.py        # PyTorch LSTM deep learning model for sequencing
rag_pipeline.py          # Document intelligence using LangChain, FAISS & Ollama
main.py                  # FastAPI application entry point
Dockerfile               # Docker configuration for containerization
requirements.txt         # Python dependencies
```

---

## ⚙️ Prerequisites (CRITICAL)

Before running this project, ensure you have the following installed on your host machine:

1. **Docker & Docker Desktop** — To run the containerized API.

2. **Ollama** — The RAG pipeline requires local LLM execution.
   - Download Ollama from [ollama.com](https://ollama.com)
   - Run this command in your terminal to download the Llama 3 model:
     ```bash
     ollama run llama3
     ```

3. **SEC API Key** *(Optional but Recommended)* — For fetching official 10-K filings.
   - Create a `.env` file in the root directory.
   - Add the following line:
     ```
     SEC_API_KEY=your_api_key_here
     ```

---

## 🚀 Installation & Running

**1. Clone the repository:**
```bash
git clone <repository-url>
cd Finansoer_Test
```

**2. Build the Docker Image:**
```bash
docker build -t finansoer .
```

**3. Run the Docker Container:**
```bash
docker run -p 8000:8000 --env-file .env finansoer
```

> **Note:** The container is configured to communicate with your host machine's Ollama instance via `host.docker.internal`.

---

## 🧪 Usage & Testing the API

Once the container is running, the FastAPI server will be active.

### Method 1: Interactive API Docs (Recommended)

Open your browser and navigate to the built-in Swagger UI:

👉 **http://localhost:8000/docs**

From here you can:
- Click on the `GET /analyze/{ticker}` endpoint
- Click **"Try it out"**
- Enter a stock symbol (e.g., `AAPL`, `NVDA`, `TSLA`)
- Hit **Execute**

### Method 2: Direct URL

Trigger the pipeline directly via your browser or Postman:

👉 `http://localhost:8000/analyze/NVDA`

---

## 🔄 What Happens When You Request an Analysis?

| Step | Description |
|------|-------------|
| **Step 1** | Historical data is collected or updated. |
| **Step 2** | Technical indicators are engineered. |
| **Step 3** | The Hybrid AI Model (XGBoost + LSTM) calculates the probability of a sharp decline and generates a SHAP explanation chart (`shap_explanaiton.png`). |
| **Step 4** | Llama 3 acts as a quantitative analyst, reading SEC filings/news and justifying the calculated numerical risk score in a professional summary. |
