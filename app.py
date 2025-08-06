import os
import uuid
import streamlit as st
from fpdf import FPDF
import fitz  # PyMuPDF
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain.chains import RetrievalQA, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain import hub

# --- 1. Configuration (from api_rag.py) ---
# Use st.secrets for your API key on Streamlit Cloud
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "YOUR_DEFAULT_API_KEY_HERE")
FAISS_INDEX_PATH = "oxford_handbook_kb"
TEMP_STORAGE_PATH = "temp_user_docs"
CHEATSHEET_PATH = "downloads"

os.makedirs(TEMP_STORAGE_PATH, exist_ok=True)
os.makedirs(CHEATSHEET_PATH, exist_ok=True)

# --- Backend Components (from api_rag.py) ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY)

# --- Helper Function for PDFs (from api_rag.py) ---
def create_formatted_pdf(text_content: str, topic: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    try:
        # Assumes fonts are in the root directory
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf", uni=True)
    except RuntimeError:
        # FPDF raises RuntimeError if font file is not found
        st.error("Could not find 'DejaVuSans.ttf' or 'DejaVuSans-Bold.ttf'. Please ensure they are in the root folder.")
        return ""

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

# --- Main Query Logic (Integrated from api_rag.py) ---
def handle_query_logic(query: str, session_id: str = None):
    # Step 1: Select the correct retriever
    if session_id:
        temp_db_path = os.path.join(TEMP_STORAGE_PATH, session_id)
        if not os.path.exists(temp_db_path):
            return "Error: Your document session has expired or is invalid. Please upload the document again.", None
        db = FAISS.load_local(temp_db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        if not os.path.exists(FAISS_INDEX_PATH):
             return "Error: The default knowledge base is not available. Please upload a document to begin.", None
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Step 2: Define the tools
    @tool
    def question_answer_tool(query: str) -> str:
        """Use this tool ONLY to answer a direct, specific question from the user."""
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        return chain.invoke(query)['result']

    @tool
    def concept_explainer_tool(topic: str) -> str:
        """Use this tool ONLY when the user asks to be taught or for a detailed explanation of a topic."""
        context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(topic)])
        prompt = PromptTemplate.from_template("Provide a comprehensive explanation of {topic}.\n\nContext: {context}\nLecture:")
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(topic=topic, context=context)

    @tool
    def cheatsheet_generator_tool(topic: str) -> str:
        """Use this tool ONLY when the user asks for a cheat sheet, summary, or key points. This tool generates a well-formatted PDF."""
        context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(topic)])
        prompt = PromptTemplate.from_template("Create a detailed cheat sheet for {topic} using '##' for headings and '-' for list items.\nContext: {context}\nCheat Sheet:")
        chain = LLMChain(llm=llm, prompt=prompt)
        cheatsheet_text = chain.run(topic=topic, context=context)
        pdf_filename = create_formatted_pdf(cheatsheet_text, topic)
        return f"PDF_GENERATED::{pdf_filename}::{cheatsheet_text}"

    tools = [question_answer_tool, concept_explainer_tool, cheatsheet_generator_tool]
    
    # Step 3: Run the agent
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True, return_intermediate_steps=True)
    response = agent_executor.invoke({"input": query})
    
    final_answer = response.get('output', "I couldn't find an answer.")
    pdf_filename = None

    # Step 4: Process results for PDF
    if 'intermediate_steps' in response:
        for _, observation in response['intermediate_steps']:
            if isinstance(observation, str) and observation.startswith("PDF_GENERATED::"):
                try:
                    parts = observation.split("::")
                    pdf_filename = parts[1]
                except IndexError:
                    pass
    return final_answer, pdf_filename

# --- Streamlit UI (from api_front.py) ---
st.set_page_config(layout="centered")

# --- Theme Dictionaries & Session State from original UI ---
LIGHT = { "bg": "#f8fafb", "bar": "#fff", "bot": "#e9eef6", "user": "#d1e7dd", "text": "#191b22", "input": "#e8edf2", "border": "#d4dde7", "expander": "#f4f7fb" }
DARK = { "bg": "#18181c", "bar": "#202126", "bot": "#232733", "user": "#22577a", "text": "#f3f5f8", "input": "#242730", "border": "#26282f", "expander": "#24272e" }

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if "active_doc_name" not in st.session_state:
    st.session_state["active_doc_name"] = None

THEME = DARK if st.session_state["theme"] == "dark" else LIGHT

# --- Custom CSS Styling from original UI ---
st.markdown(f"""
<style>
    .stApp {{
        background-color: {THEME['bg']};
        color: {THEME['text']};
    }}
    [data-testid="stSidebar"] {{
        background-color: {THEME['bar']};
    }}
    .stChatMessage {{
        background-color: {THEME['bot']};
        border-radius: 0.5rem;
    }}
    div[data-testid="stChatMessage"][data-user-border-color] {{
        background-color: {THEME['user']};
    }}
    [data-testid="stChatInput"] {{
        background-color: {THEME['bg']};
    }}
     [data-testid="stChatInput"] textarea {{
        background-color: {THEME['input']};
        color: {THEME['text']};
    }}
    .stButton>button, .stDownloadButton>button {{
        border: 1px solid {THEME['border']};
        background-color: {THEME['input']};
        color: {THEME['text']};
    }}
</style>
""", unsafe_allow_html=True)

# --- Top Bar with Title and Theme Toggle (from original UI) ---
def top_bar():
    col1, col2 = st.columns([10, 1])
    with col1:
        st.title("Ophthalmology Learning Assistant")
   if st.button("☀️", key="theme-sun", help="Switch to light mode"): st.session_state["theme"] = "light"; st.rerun()
        if st.button("🌙", key="theme-moon", help="Switch to dark mode"): st.session_state["theme"] = "dark"; st.rerun()

top_bar()

# --- Sidebar for document upload (integrating backend logic) ---
with st.sidebar:
    st.header("Upload Your Document")
    uploaded_file = st.file_uploader("Upload a PDF to ask questions about it", type="pdf")
    
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document... This may take a moment."):
                # This logic is from the /upload endpoint in api_rag.py
                session_id = str(uuid.uuid4())
                temp_dir = os.path.join(TEMP_STORAGE_PATH, session_id)
                os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as buffer:
                    buffer.write(uploaded_file.getbuffer())
                
                doc = fitz.open(file_path)
                full_text = "".join(page.get_text() for page in doc)
                doc.close()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                texts = text_splitter.split_text(full_text)
                
                temp_db = FAISS.from_texts(texts, embeddings)
                temp_db.save_local(temp_dir)
                
                st.session_state["session_id"] = session_id
                st.session_state["active_doc_name"] = uploaded_file.name
                st.session_state["chat_history"] = [] # Clear history for new doc
                st.success(f"Processed '{uploaded_file.name}'")
    
    st.divider()

    if st.session_state["active_doc_name"]:
        st.info(f"Active document: **{st.session_state['active_doc_name']}**")
    else:
        st.info("Using the default Ophthalmology knowledge base.")

# --- Chat interface (integrating backend logic) ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "pdf_filename" in message and message["pdf_filename"]:
            pdf_path = os.path.join(CHEATSHEET_PATH, message["pdf_filename"])
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="Download Cheatsheet",
                        data=pdf_file,
                        file_name=message["pdf_filename"],
                        mime='application/pdf'
                    )

if prompt := st.chat_input("Ask a question, request a topic explanation, or ask for a cheat sheet..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        # This logic is from the /query endpoint in api_rag.py
        answer, pdf_filename = handle_query_logic(prompt, st.session_state.get("session_id"))
        
        with st.chat_message("assistant"):
            st.markdown(answer)
            if pdf_filename:
                pdf_path = os.path.join(CHEATSHEET_PATH, pdf_filename)
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download Cheatsheet",
                            data=pdf_file,
                            file_name=pdf_filename,
                            mime='application/pdf'
                        )
        
        # Append the complete response, including pdf_filename for re-rendering
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "pdf_filename": pdf_filename})

