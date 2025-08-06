import os
import uuid
import streamlit as st
import speech_recognition as sr
from fpdf import FPDF
import fitz # PyMuPDF
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain.chains import RetrievalQA, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain import hub

# --- 1. Configuration and Setup ---

# IMPORTANT: Set your GOOGLE_API_KEY in Streamlit's secrets
# Go to your app's settings > secrets and add GOOGLE_API_KEY = "your_key_here"
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

FAISS_INDEX_PATH = "oxford_handbook_kb"
CHEATSHEET_PATH = "downloads"

# Create directories if they don't exist
os.makedirs(CHEATSHEET_PATH, exist_ok=True)

# --- 2. Core Application Logic (from your backend) ---

# Initialize LLM and Embeddings
# Add a check to ensure the API key is available
if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set it in Streamlit's secrets.")
    st.stop()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY)

# PDF Generation Function
def create_formatted_pdf(text_content: str, topic: str) -> str:
    # This function is unchanged from your previous version
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Cheatsheet: {topic.title()}", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, text_content)
    filename = f"{uuid.uuid4()}.pdf"
    filepath = os.path.join(CHEATSHEET_PATH, filename)
    pdf.output(filepath)
    return filepath

# Load the default knowledge base once and cache it
@st.cache_resource
def load_default_db():
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

default_db = load_default_db()

# --- 3. Streamlit User Interface ---

# Page configuration
st.set_page_config(layout="centered", page_title="Ophthalmology AI")

# UI Styling (your CSS here)
st.markdown("""<style> ... </style>""", unsafe_allow_html=True) # Keep your full CSS

# Top Bar
st.markdown("<div class='topbar-custom'>Ophthalmology AI Assistant</div>", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None

# Chat History Display
for entry in st.session_state.chat_history:
    if "user" in entry:
        st.chat_message("user").write(entry["user"])
    else:
        st.chat_message("assistant").write(entry["bot"])
        if entry.get("pdf_path"):
            with open(entry["pdf_path"], "rb") as f:
                st.download_button("📥 Download Cheatsheet", f, file_name=os.path.basename(entry["pdf_path"]))

# Document Upload and Processing
with st.expander("Upload a Custom Document"):
    uploaded_file = st.file_uploader("Upload a PDF to ask questions about it", type="pdf")
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                # All processing now happens directly in the Streamlit app
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                full_text = "".join(page.get_text() for page in doc)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                texts = text_splitter.split_text(full_text)
                
                # Create a temporary in-memory vector store for the session
                st.session_state.custom_db = FAISS.from_texts(texts, embeddings)
                st.session_state.active_doc_name = uploaded_file.name
                st.chat_message("assistant").write(f"Ready for questions about **{uploaded_file.name}**.")

# Handle the query from text or voice input
prompt = st.chat_input("Type your question here...")

if prompt:
    st.session_state.chat_history.append({"user": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking..."):
        # Determine which retriever to use
        if 'custom_db' in st.session_state and st.session_state.custom_db:
            retriever = st.session_state.custom_db.as_retriever()
        elif default_db:
            retriever = default_db.as_retriever()
        else:
            st.error("Knowledge base not found.")
            st.stop()

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
            prompt_template = PromptTemplate.from_template("Create a cheat sheet for {topic} based on this context: {context}")
            chain = LLMChain(llm=llm, prompt=prompt_template)
            cheatsheet_text = chain.run(topic=topic, context=context)
            pdf_path = create_formatted_pdf(cheatsheet_text, topic)
            return f"PDF_GENERATED::{pdf_path}::{cheatsheet_text}"

        # Run the agent
        tools = [question_answer_tool, cheatsheet_generator_tool]
        agent_prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, agent_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)
        
        response = agent_executor.invoke({"input": prompt})
        
        final_answer = response.get('output', "Sorry, I couldn't find an answer.")
        pdf_path = None
        if 'intermediate_steps' in response:
            for _, observation in response['intermediate_steps']:
                if isinstance(observation, str) and observation.startswith("PDF_GENERATED::"):
                    pdf_path = observation.split("::")[1]
                    break

        bot_message = {"bot": final_answer, "pdf_path": pdf_path}
        st.session_state.chat_history.append(bot_message)
        st.rerun()

