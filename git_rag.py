import os
import uuid
from fpdf import FPDF
import fitz # PyMuPDF
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain.chains import RetrievalQA, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain import hub
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# --- Configuration (MODIFIED for Deployment) ---
# Load the API key securely from an environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FAISS_INDEX_PATH = "oxford_handbook_kb"
TEMP_STORAGE_PATH = "temp_user_docs"
CHEATSHEET_PATH = "downloads"

os.makedirs(TEMP_STORAGE_PATH, exist_ok=True)
os.makedirs(CHEATSHEET_PATH, exist_ok=True)

# --- Pydantic models ---
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    disclaimer: str
    pdf_url: Optional[str] = None

# --- FastAPI app & Core Components ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

if not GOOGLE_API_KEY:
    raise ValueError("FATAL ERROR: GOOGLE_API_KEY environment variable is not set.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY)

# --- Helper Function for PDFs ---
def create_formatted_pdf(text_content: str, topic: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf", uni=True)
    except RuntimeError:
        print("DejaVu fonts not found. Falling back to standard fonts.")
        pdf.add_font("Arial", "", "Arial.ttf", uni=True)
        pdf.set_font("Arial", "", 12)
    
    pdf.set_font("DejaVu", "B", 18)
    pdf.cell(0, 10, f"Ophthalmology Cheatsheet: {topic.title()}", 0, 1, 'C')
    pdf.ln(10)

    for line in text_content.split('\n'):
        line = line.strip()
        if line.startswith('## '):
            pdf.set_font("DejaVu", "B", 14)
            pdf.cell(0, 10, line.replace('## ', ''), 0, 1, 'L')
            pdf.ln(2)
        elif line.startswith('- '):
            pdf.set_font("DejaVu", "", 11)
            pdf.set_x(15)
            pdf.multi_cell(0, 7, f"• {line.replace('- ', '')}")
            pdf.ln(1)
        else:
            pdf.set_font("DejaVu", "", 11)
            pdf.multi_cell(0, 7, line)
            pdf.ln(2)
            
    unique_id = uuid.uuid4()
    filename = f"{unique_id}.pdf"
    filepath = os.path.join(CHEATSHEET_PATH, filename)
    pdf.output(filepath)
    return filename

# --- Upload endpoint ---
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    temp_dir = os.path.join(TEMP_STORAGE_PATH, session_id)
    os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    doc = fitz.open(file_path)
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(full_text)
    temp_db = FAISS.from_texts(texts, embeddings)
    temp_db.save_local(temp_dir)
    return {"message": "File processed successfully", "session_id": session_id}

# --- Main Query Endpoint (Now with robust error handling) ---
@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    try:
        print(f"Received query: '{request.query}' for session: {request.session_id}")
        
        # Determine the retriever
        if request.session_id:
            db_path = os.path.join(TEMP_STORAGE_PATH, request.session_id)
            if not os.path.exists(db_path):
                return QueryResponse(answer="Error: Your document session has expired.", disclaimer="")
            db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        else:
            db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Define tools using the selected retriever
        @tool
        def question_answer_tool(query: str) -> str:
            """Answers a direct, specific question."""
            chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            return chain.invoke(query)['result']

        @tool
        def cheatsheet_generator_tool(topic: str) -> str:
            """Creates a summary or cheat sheet PDF."""
            context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(topic)])
            prompt = PromptTemplate.from_template("Create a cheat sheet for {topic} using '##' for headings and '-' for list items.\nContext: {context}\nCheat Sheet:")
            chain = LLMChain(llm=llm, prompt=prompt)
            cheatsheet_text = chain.run(topic=topic, context=context)
            pdf_filename = create_formatted_pdf(cheatsheet_text, topic)
            return f"PDF_GENERATED::{pdf_filename}::{cheatsheet_text}"

        tools = [question_answer_tool, cheatsheet_generator_tool]
        
        # Run agent
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)
        
        response = agent_executor.invoke({"input": request.query})
        final_answer = response.get('output', "The agent did not provide a final answer.")
        pdf_url = None

        if 'intermediate_steps' in response:
            for _, observation in response['intermediate_steps']:
                if isinstance(observation, str) and observation.startswith("PDF_GENERATED::"):
                    filename = observation.split("::")[1]
                    pdf_url = f"/download/{filename}" # Use a relative path
                    break
        
        return QueryResponse(answer=final_answer, disclaimer="For educational purposes only.", pdf_url=pdf_url)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        error_message = f"**An internal error occurred in the backend.**\n\n**Details:** {str(e)}"
        return QueryResponse(answer=error_message, disclaimer="Please check the server logs.", pdf_url=None)

# --- Download endpoint ---
@app.get("/download/{filename}")
async def download_pdf(filename: str):
    file_path = os.path.join(CHEATSHEET_PATH, filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, media_type='application/pdf', filename=filename)
    return {"error": "File not found"}
