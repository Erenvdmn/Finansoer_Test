import os
import yfinance as yf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


class DocumentIntelligence: 
    def __init__(self):
        self.llm = OllamaLLM(model="llama3")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def fetch_real_financial_data(self, ticker):
        print(f"{ticker}'s financial datas pulling with Yahoo Finance...")
        stock = yf.Ticker(ticker)

        # Business Summary
        company_info = stock.info.get('longBusinessSummary', 'Infos could not found')

        # Last 5 news about companies 
        news_items = stock.news
        news_text = "Recent Market News:\n"

        for item in news_items[:5]:
            title = item.get('title', 'Title could not get')
            publisher = item.get('publisher', 'Unknown resource')
            news_text += f"- {title} (Source: {publisher})\n"

        full_text = f"Company Context:\n{company_info}\n\n{news_text}"

        print(f"Data pulled sucsessfully, Llama 3 train with news right now")
        return [Document(page_content=full_text)]
    

    def build_vector_database(self, ticker):
        docs = self.fetch_real_financial_data(ticker)

        print("Datas chunking...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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
        You are a senior Wall Street quantitative analyst. 
        Our XGBoost machine learning model calculated a risk score of {numerical_risk_score}% for {ticker} based on technical indicators.
        
        Here is the real-time market data and recent news context for {ticker}:
        {context_text}
        
        Based ONLY on the provided context above, answer this:
        Does the real-time data reveal any hidden risks, challenges, or positive news that the numerical model missed? 
        Should investors adjust their expectations despite the {numerical_risk_score}% numerical risk?
        Explain briefly in 3-4 sentences.
        """

        print(f"\n🚀 Llama 3, thinking for {ticker} financial data...\n")
        print("-" * 50)

        response = self.llm.invoke(prompt)
        print(response)
        print("\n" + "-" * 50)

# for manuel testing of this file
if __name__=="__main__":
    rag = DocumentIntelligence()

    rag.analyze_risk(ticker="AAPL", numerical_risk_score=14.80)