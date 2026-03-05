import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.documents import Document


class DocumentIntelligence:
    def __init__(self):
        self.llm = Ollama(model="llama3")

        # Model that will turn texts to vectors
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def create_mock_sec_report(self):
        print("Apple SEC (10-K) Risk Raporu preparing...")
        report_text = """
        Item 1A. Risk Factors (Apple Inc.)
        We face significant supply chain constraints in our primary manufacturing hubs due to recent geopolitical tensions. 
        Additionally, the European Union's new regulatory frameworks (Digital Markets Act) force us to allow third-party app stores, 
        which is expected to severely reduce our Service sector revenue margins in the upcoming quarters. 
        Furthermore, global consumer demand for premium smartphones is showing signs of cooling off amidst rising inflation. 
        While our balance sheet remains strong, these macroeconomic headwinds pose a material risk to short-term stock performance.
        """

        return [Document(page_content=report_text)]
    
    def build_vector_database(self):
        docs = self.create_mock_sec_report()

        # Chunking
        print("Texts is Chunking...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        print("Creating FAISS...")
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return vectorstore
    
    def analyze_risk(self, numerical_risk_score):
        vectorstore = self.build_vector_database()

        retriever=vectorstore.as_retriever()

        relevant_docs = retriever.invoke("What are the macroeconomic and supply chain risks?")
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
        You are a senior Wall Street quantitative analyst. 
        Our XGBoost machine learning model calculated a sharp decline risk score of {numerical_risk_score}% for Apple (AAPL) based on technical indicators like SMA_50 and Volatility.
        
        Here is the SEC risk report context:
        {context_text}
        
        Based ONLY on the provided SEC risk report context above, answer this:
        Does the text reveal any hidden risks that the numerical model missed? Should investors be worried despite the low {numerical_risk_score}% numerical risk?
        Explain briefly in 3-4 sentences.
        """

        print("Llama 3 is thinking and writing report...")
        print("-"*50)

        response = self.llm.invoke(prompt)
        print(response)
        print("\n"+"-"*50)

if __name__=="__main__":
    rag = DocumentIntelligence()
    rag.analyze_risk(numerical_risk_score=14.80)