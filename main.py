from collecting_data import DataDownloader
from risk_scoring import RiskScorer
from rag_pipeline import DocumentIntelligence


def run_fintech_pipeline(ticker):
    print(f"\n{'='*50}")
    print(f"CREATING ANALYZE for {ticker}...")
    print(f"\n{'='*50}")


    print("[STEP ONE] Pulling financial data...")
    downloader = DataDownloader()
    downloader.get_daily_data(ticker)

    print("[STEP TWO] Machine Learning (XGBoost) Analizing Risk Score...")
    filepath= f"data/{ticker}_daily.csv"
    scorer = RiskScorer(filepath)
    calculated_risk_score = scorer.train_and_explain()

    if calculated_risk_score is None:
        print("Risk score could not calculated")
        return
    
    print("[STEP THREE] Llama 3 (RAG) Reading Current News...")
    rag = DocumentIntelligence()
    rag.analyze_risk(ticker=ticker, numerical_risk_score=round(calculated_risk_score, 2))

    print(f"\n{'='*50}")
    print(f"✅ {ticker} ANALYZE COMPLATED!")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    POINTED_TICKER = "TSLA" 
    
    run_fintech_pipeline(POINTED_TICKER)