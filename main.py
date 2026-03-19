from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import os

from collecting_data import DataDownloader
from feature_engineering import FeatureEngineer
from risk_scoring import RiskScorer
from rag_pipeline import DocumentIntelligence

app = FastAPI(
    title="FINANSOER - AI Based Finans Assistant",
    description="Hybrid ML/DL and LLM-powered SEC Edgar Financial Analysis API",
    version="1.0.0"
)

class AnalyzeResponse(BaseModel):
    ticker: str
    risk_score: float
    ai_analysis: str


@app.get("/")
def read_root():
    return {"message": "🚀 Corparete GenAI Model is Active! Please use /analyze/{ticker} endpoint."}



@app.get("/analyze/{ticker}", response_model=AnalyzeResponse)
def analyze_stock(ticker: str):
    ticker = ticker.upper()
    print(f"\n{'='*50}")
    print(f"🌐 GOT API REQUEST {ticker}'s Analyze Starting...")
    print(f"{'='*50}\n")

    try:
        # Pulling Data
        print(f"[{ticker}] Step 1: Tickers Data are Collecting/Updating...")
        downloader = DataDownloader()
        downloader.get_daily_data(ticker)
        
        filepath = f"data/{ticker}_daily.csv"
        # Feature Engineering
        print(f"[{ticker}] Step 2: Preparing data...")
        engineer = FeatureEngineer(filepath)
        engineer.get_processed_data() 
        
        # --- ADIM 3: Hybrid Risk Score (XGBoost + LSTM)
        print(f"[{ticker}] Adım 3: Derin Öğrenme ve Makine Öğrenmesi Modelleri Çalışıyor...")
        scorer = RiskScorer(filepath)
        risk_score = scorer.train_and_explain()
        
        if risk_score is None:
            raise HTTPException(status_code=400, detail="Couldn't calculate risk score")

        # --- ADIM 4: SEC EDGAR RAG (Llama 3) ---
        print(f"[{ticker}] Adım 4: Reading SEC 10-K Docs...")
        rag = DocumentIntelligence()
        ai_analysis = rag.analyze_risk(ticker=ticker, numerical_risk_score=round(risk_score, 2))

        print(f"\n✅ {ticker} ANALYZE TAMAMLANDICOMPLETED! SENDING JSON RESULT...")
        
        return AnalyzeResponse(
            ticker=ticker,
            risk_score=round(risk_score, 2),
            ai_analysis=ai_analysis
        )

    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    IP_Adress="192.168.1.107"
    uvicorn.run("main:app", host={IP_Adress}, port=8000, reload=False)