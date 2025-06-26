from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain.agents import initialize_agent, Tool
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

from salesforce_utils import connect_salesforce, create_opportunity

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Gemini LLM & Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Vectorstore
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Tools
qa_tool = Tool(
    name="TrainingGuideQA",
    func=qa_chain.run,
    description="Answer Salesforce training questions."
)

def create_opportunity_tool(input_str: str):
    try:
        name, stage, close_date, amount = map(str.strip, input_str.split(','))
        sf = connect_salesforce()
        result = create_opportunity(sf, name, stage, close_date, float(amount))
        return f"✅ Opportunity created: {result['id']}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

opportunity_tool = Tool(
    name="CreateOpportunity",
    func=create_opportunity_tool,
    description="Create opportunity. Format: Name,Stage,CloseDate,Amount"
)

# Agent
agent = initialize_agent(
    tools=[qa_tool, opportunity_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Request model
class ChatRequest(BaseModel):
    message: str

# Endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    result = agent.run(request.message)
    return {"response": result}
