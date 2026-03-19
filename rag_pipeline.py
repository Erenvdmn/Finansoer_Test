import os
import yfinance as yf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

SEC_API_KEY = os.getenv("SEC_API_KEY")


try:
    from sec_api import QueryApi, ExtractorApi
    SEC_API_AVAILABLE = True
except ImportError:
    SEC_API_AVAILABLE = False

class DocumentIntelligence: 
    def __init__(self, sec_api_key = SEC_API_KEY):
        self.llm = OllamaLLM(model="llama3", base_url="http://host.docker.internal:11434")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.sec_api_key = sec_api_key

    def fetch_real_financial_data(self, ticker):
        print(f"{ticker}'s financial datas pulling with Yahoo Finance...")
        stock = yf.Ticker(ticker)

        # Business Summary
        company_info = stock.info.get('longBusinessSummary', 'Infos could not found')

        risk_factors_text =""

        if SEC_API_AVAILABLE and self.sec_api_key:
            try:
                print("🔍 Searching for the latest 10-K filing on SEC EDGAR...")
                queryApi = QueryApi(api_key=self.sec_api_key)
                
                query = {
                  "query": { "query_string": { "query": f"ticker:{ticker} AND formType:\"10-K\"" } },
                  "from": "0",
                  "size": "1",
                  "sort": [{ "filedAt": { "order": "desc" } }]
                }
                response = queryApi.get_filings(query)
                
                if response['filings']:
                    filing_url = response['filings'][0]['linkToFilingDetails']
                    print(f"📄 Found 10-K URL: {filing_url}")
                    print("⏳ Extracting 'Item 1A: Risk Factors' (This may take a few seconds)...")
                    
                    extractorApi = ExtractorApi(api_key=self.sec_api_key)
                    item_1a = extractorApi.get_section(filing_url, "1A", "text")
                    
                    risk_factors_text = item_1a[:10000] 
                    print("✅ Item 1A Extracted Successfully!")
                else:
                    print("⚠️ No 10-K filings found for this ticker on SEC.")
            except Exception as e:
                print(f"❌ SEC API Error: {e}")

        if not risk_factors_text:
            print("⚠️ SEC API Key missing or error occurred. Using Yahoo Finance recent news as fallback.")
            news_items = stock.news
            news_text = "Recent Market News:\n"
            for item in news_items[:5]:
                title = item.get('title', 'Title could not get')
                publisher = item.get('publisher', 'Unknown resource')
                news_text += f"- {title} (Source: {publisher})\n"
            risk_factors_text = news_text

        full_text = f"Company Context:\n{company_info}\n\nOfficial Risk Factors / News:\n{risk_factors_text}"
        
        print("Data pulled successfully, vectorizing for Llama 3...")
        return [Document(page_content=full_text, metadata={"source": "hybrid_sec_yahoo"})]

    def build_vector_database(self, ticker):
        docs = self.fetch_real_financial_data(ticker)

        print("Datas chunking...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
        chunks = text_splitter.split_documents(docs)

        print("Vector Database (FAISS) creating...")
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return vectorstore
    
    def analyze_risk(self, ticker, numerical_risk_score):
        vectorstore = self.build_vector_database(ticker)
        retriever = vectorstore.as_retriever()

        relevant_docs = retriever.invoke("What are the recent news, challenges or risks?")
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
        You are a Senior Wall Street Quantitative Analyst. 
        Our XGBoost/LSTM hybrid model calculated an imminent sharp decline risk score of {numerical_risk_score}% for {ticker}.
        
        Here is the official SEC 10-K Risk Factors (Item 1A) and business context for {ticker}:
        {context_text}
        
        Based ONLY on the provided SEC text above, answer this:
        1. Are there specific operational, legal, or supply chain risks in the text that justify this {numerical_risk_score}% numerical risk?
        2. Should investors adjust their expectations based on these fundamental SEC risk disclosures?
        
        Give a highly professional, brief, and analytical response in 3-4 sentences. Do not use generic filler words.
        """

        print(f"\n🚀 Llama 3, thinking for {ticker} financial data...\n")
        print("-" * 50)

        response = self.llm.invoke(prompt)
        print(response)
        print("\n" + "-" * 50)

        return response

# for manuel testing of this file
if __name__=="__main__":
    rag = DocumentIntelligence()

    rag.analyze_risk(ticker="AAPL", numerical_risk_score=14.80)