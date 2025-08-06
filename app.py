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

# --- 1. Configuration ---
# Use st.secrets for your API key on Streamlit Cloud
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "YOUR_DEFAULT_API_KEY_HERE")
FAISS_INDEX_PATH = "oxford_handbook_kb"
TEMP_STORAGE_PATH = "temp_user_docs"
CHEATSHEET_PATH = "downloads"

os.makedirs(TEMP_STORAGE_PATH, exist_ok=True)
os.makedirs(CHEATSHEET_PATH, exist_ok=True)

# --- Backend Components ---
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

# --- Main Query Logic (Integrated from backend) ---
def handle_query_logic(query: str, session_id: str = None):
    # Step 1: Select retriever
    if session_id:
        temp_db_path = os.path.join(TEMP_STORAGE_PATH, session_id)
        if not os.path.exists(temp_db_path):
            return "Error: Your document session has expired. Please upload the document again.", None
        db = FAISS.load_local(temp_db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        if not os.path.exists(FAISS_INDEX_PATH):
             return "Error: The default knowledge base is not available. Upload a document to begin.", None
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Step 2: Define tools
    @tool
    def question_answer_tool(query: str) -> str:
        """Use for direct questions."""
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        return chain.invoke(query)['result']

    @tool
    def concept_explainer_tool(topic: str) -> str:
        """Use for detailed topic explanations."""
        context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(topic)])
        prompt = PromptTemplate.from_template("Provide a comprehensive explanation of {topic}.\n\nContext: {context}\nLecture:")
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(topic=topic, context=context)

    @tool
    def cheatsheet_generator_tool(topic: str) -> str:
        """Use for cheat sheets or summaries. Generates a PDF."""
        context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(topic)])
        prompt = PromptTemplate.from_template("Create a detailed cheat sheet for {topic} using '##' for headings and '-' for list items.\nContext: {context}\nCheat Sheet:")
        chain = LLMChain(llm=llm, prompt=prompt)
        cheatsheet_text = chain.run(topic=topic, context=context)
        pdf_filename = create_formatted_pdf(cheatsheet_text, topic)
        return f"PDF_GENERATED::{pdf_filename}::{cheatsheet_text}"

    tools = [question_answer_tool, concept_explainer_tool, cheatsheet_generator_tool]
    
    # Step 3: Run agent
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True, return_intermediate_steps=True)
    response = agent_executor.invoke({"input": query})
    
    final_answer = response.get('output', "I couldn't find an answer.")
    pdf_filename = None

    # Step 4: Extract PDF filename if generated
    if 'intermediate_steps' in response:
        for _, observation in response['intermediate_steps']:
            if isinstance(observation, str) and observation.startswith("PDF_GENERATED::"):
                try:
                    pdf_filename = observation.split("::")[1]
                except IndexError:
                    pass
    return final_answer, pdf_filename

# --- Streamlit UI (with Responsive Improvements) ---
st.set_page_config(layout="centered")

# --- Theme Dictionaries & Session State ---
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

# --- Custom CSS Styling (with Responsive Media Queries) ---
st.markdown(f"""
<style>
    .stApp {{
        background: {THEME['bg']};
        color: {THEME['text']};
    }}
    .topbar-custom {{
        background: {THEME['bar']};
        border-radius: 16px; /* Moved from inline style for consistency */
        padding: 1.3em 1.2em 1.15em 2.1em;
        margin-bottom: 1.6em;
        box-shadow: 0 2px 12px 0 rgba(44,46,66,0.06);
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: .02em;
    }}
    .msg-user {{
        background: {THEME['user']}; color: {THEME['text']}; border-radius: 16px 16px 4px 20px;
        margin-bottom: 0.3em; padding: 1em 1.35em; width: fit-content; max-width: 85%;
        font-size: 1.13rem; border: 1.5px solid {THEME['border']}; margin-left: auto;
        margin-right: 0; text-align: right; box-shadow: 0 1px 12px 0 rgba(55,96,148,0.05);
        word-break: break-word;
    }}
    .msg-bot {{
        background: {THEME['bot']}; color: {THEME['text']}; border-radius: 16px 16px 20px 4px;
        margin-bottom: 0.7em; padding: 1.08em 1.23em 1em 1.18em; width: fit-content;
        max-width: 85%; font-size: 1.13rem; border: 1.5px solid {THEME['border']};
        margin-right: auto; margin-left: 0; text-align: left;
        box-shadow: 0 1px 12px 0 rgba(44,46,66,0.05); word-break: break-word;
    }}
    [data-testid="stExpander"] {{
        border-color: {THEME['border']};
        background: {THEME['expander']};
    }}
    .stButton>button, .stDownloadButton>button {{
        border: 1px solid {THEME['border']};
    }}

    /* --- RESPONSIVE STYLES FOR MOBILE --- */
    @media (max-width: 768px) {{
        .topbar-custom {{
            font-size: 1.2rem; /* Smaller font for title on mobile */
            padding: 1em;      /* Reduced padding */
            text-align: center;
        }}
        .msg-user, .msg-bot {{
            font-size: 0.95rem;  /* Smaller font for chat messages */
            max-width: 90%;    /* Allow slightly wider messages */
        }}
    }}
</style>
""", unsafe_allow_html=True)

# --- Top bar with theme buttons ---
def top_bar():
    col1, col2, col3 = st.columns([8, 1, 1]) # Adjusted column ratio
    with col1:
        # Removed inline style, now controlled by CSS class
        st.markdown("<div class='topbar-custom'>Ophthalmology AI Assistant</div>", unsafe_allow_html=True)
    with col2:
        # Added use_container_width for better mobile layout
        if st.button("☀️", key="theme-sun", help="Switch to light mode"): st.session_state["theme"] = "light"; st.rerun()
        if st.button("🌙", key="theme-moon", help="Switch to dark mode"): st.session_state["theme"] = "dark"; st.rerun()

top_bar()

# --- Chat History Display ---
for entry in st.session_state.chat_history:
    if "user" in entry:
        st.markdown(f"<div class='msg-user'>{entry['user']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='msg-bot'>{entry['bot']}</div>", unsafe_allow_html=True)
        if entry.get("pdf_filename"):
            pdf_path = os.path.join(CHEATSHEET_PATH, entry["pdf_filename"])
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button("📥 Download Cheatsheet", pdf_file.read(), entry["pdf_filename"], "application/pdf")

# --- Document Upload Expander ---
with st.expander("Upload a Custom Document"):
    uploaded_file = st.file_uploader("Upload a PDF to ask questions about it", type="pdf")
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
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
                st.session_state.chat_history.append({"bot": f"Ready for questions about **{uploaded_file.name}**."})
                st.rerun()

# --- Active Document Status ---
if st.session_state["active_doc_name"]:
    st.info(f"Active Document: **{st.session_state['active_doc_name']}**")
    if st.button("Clear Document & Revert to Default"):
        st.session_state["session_id"] = None
        st.session_state["active_doc_name"] = None
        st.session_state.chat_history.append({"bot": "Reverted to default knowledge base."})
        st.rerun()

# --- User Input & Logic ---
if user_prompt := st.chat_input("Type your question here..."):
    st.session_state.chat_history.append({"user": user_prompt})
    
    with st.spinner("Thinking..."):
        answer, pdf_filename = handle_query_logic(user_prompt, st.session_state.get("session_id"))
        st.session_state.chat_history.append({"bot": answer, "pdf_filename": pdf_filename})
    st.rerun()
