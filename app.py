# Place this at the very beginning of your Streamlit script
import os
import uuid
import fitz
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF
from langchain.chains import RetrievalQA, LLMChain
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain import hub

st.set_page_config(layout="centered")

# --- 1. Device Width Detection ---
if "screen_width" not in st.session_state:
    st.session_state["screen_width"] = 768
    components.html("""
        <script>
        const sendWidth = () => {
            const width = window.innerWidth;
            const streamlitDoc = window.parent.document;
            const data = {"width": width};
            streamlitDoc.dispatchEvent(new CustomEvent("streamlit:screen-width", {detail: data}));
        };
        window.addEventListener("resize", sendWidth);
        sendWidth();
        </script>
    """, height=0, scrolling=False)

# --- 2. Determine Device Type ---
def get_device_type():
    try:
        width = int(st.session_state.get("screen_width", 768))
    except (ValueError, TypeError):
        width = 768
    if width < 576:
        return "mobile"
    elif width < 992:
        return "tablet"
    else:
        return "desktop"

DEVICE = get_device_type()

# --- 3. Theme Variables ---
LIGHT = {"bg": "#f8fafb", "bar": "#fff", "bot": "#e9eef6", "user": "#d1e7dd", "text": "#191b22", "input": "#e8edf2", "border": "#d4dde7", "expander": "#f4f7fb"}
DARK = {"bg": "#18181c", "bar": "#202126", "bot": "#232733", "user": "#22577a", "text": "#f3f5f8", "input": "#242730", "border": "#26282f", "expander": "#24272e"}

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
THEME = DARK if st.session_state["theme"] == "dark" else LIGHT

# --- 4. Responsive CSS Styling ---
msg_font = "0.95rem" if DEVICE == "mobile" else "1.1rem"
padding = "0.8rem" if DEVICE == "mobile" else "1.2rem"
topbar_font = "1.1rem" if DEVICE == "mobile" else "1.5rem"

st.markdown(f"""
<style>
html, body, .stApp {{
    max-width: 100vw;
    overflow-x: hidden;
    padding: 0;
    margin: 0;
    font-family: 'Segoe UI', sans-serif;
    background-color: {THEME['bg']};
    color: {THEME['text']};
}}

.topbar-custom {{
    background-color: {THEME['bar']};
    border-radius: 12px;
    padding: {padding};
    font-size: {topbar_font};
    font-weight: bold;
    text-align: center;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}}

.msg-user, .msg-bot {{
    max-width: 90%;
    padding: {padding};
    margin-bottom: 0.5rem;
    font-size: {msg_font};
    border-radius: 12px;
    border: 1px solid {THEME['border']};
    word-wrap: break-word;
    word-break: break-word;
    box-shadow: 0 1px 5px rgba(0,0,0,0.05);
}}

.msg-user {{
    background-color: {THEME['user']};
    margin-left: auto;
    text-align: right;
    border-radius: 12px 12px 4px 16px;
}}

.msg-bot {{
    background-color: {THEME['bot']};
    margin-right: auto;
    text-align: left;
    border-radius: 12px 12px 16px 4px;
}}

.stButton>button, .stDownloadButton>button {{
    width: 100%;
    padding: 0.6rem 1rem;
    font-size: 1rem;
    border-radius: 10px;
    margin-top: 0.5rem;
    background-color: {THEME['input']};
    border: 1px solid {THEME['border']};
}}

[data-testid="stExpander"] {{
    border: 1px solid {THEME['border']};
    background-color: {THEME['expander']};
    border-radius: 10px;
    font-size: 0.95rem;
}}

.block-container {{
    padding: 0 1rem !important;
}}
</style>
""", unsafe_allow_html=True)

# --- 5. Topbar Display ---
st.markdown(f"<div class='topbar-custom'>Ophthalmology AI Assistant ({DEVICE.title()})</div>", unsafe_allow_html=True)

# --- 6. Theme Switch Buttons ---
col1, col2 = st.columns([10, 1])
with col2:
    if st.button("☀️", key="theme-sun", help="Light mode"):
        st.session_state["theme"] = "light"
        st.rerun()
    if st.button("🌙", key="theme-moon", help="Dark mode"):
        st.session_state["theme"] = "dark"
        st.rerun()

# --- 7. Chat + Document Session State ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if "active_doc_name" not in st.session_state:
    st.session_state["active_doc_name"] = None

# --- 8. Backend Setup ---
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
FAISS_INDEX_PATH = "oxford_handbook_kb"
TEMP_STORAGE_PATH = "temp_user_docs"
CHEATSHEET_PATH = "downloads"

os.makedirs(TEMP_STORAGE_PATH, exist_ok=True)
os.makedirs(CHEATSHEET_PATH, exist_ok=True)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=GOOGLE_API_KEY)

# --- 9. Helper: PDF Generator ---
def create_formatted_pdf(text_content: str, topic: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf", uni=True)
    except RuntimeError:
        st.error("Missing font files.")
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
    filename = f"{uuid.uuid4()}.pdf"
    filepath = os.path.join(CHEATSHEET_PATH, filename)
    pdf.output(filepath)
    return filename

# --- 10. Document Upload ---
with st.expander("Upload a Custom Document"):
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing..."):
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

# --- 11. Document Info ---
if st.session_state["active_doc_name"]:
    st.info(f"Active Document: **{st.session_state['active_doc_name']}**")
    if st.button("Clear Document"):
        st.session_state["session_id"] = None
        st.session_state["active_doc_name"] = None
        st.session_state.chat_history.append({"bot": "Reverted to default knowledge base."})
        st.rerun()

# --- 12. Main Logic ---
def handle_query_logic(query: str, session_id: str = None):
    if session_id:
        db = FAISS.load_local(os.path.join(TEMP_STORAGE_PATH, session_id), embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    @tool
def question_answer_tool(query: str) -> str:
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        return chain.invoke(query)['result']

    @tool
def concept_explainer_tool(topic: str) -> str:
        context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(topic)])
        prompt = PromptTemplate.from_template("Explain {topic}.\n\nContext: {context}\nLecture:")
        return LLMChain(llm=llm, prompt=prompt).run(topic=topic, context=context)

    @tool
def cheatsheet_generator_tool(topic: str) -> str:
        context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(topic)])
        prompt = PromptTemplate.from_template("Create a cheat sheet for {topic} using '##' for headings and '-' for list items.\nContext: {context}\nCheat Sheet:")
        cheatsheet_text = LLMChain(llm=llm, prompt=prompt).run(topic=topic, context=context)
        pdf_filename = create_formatted_pdf(cheatsheet_text, topic)
        return f"PDF_GENERATED::" + pdf_filename + "::" + cheatsheet_text

    tools = [question_answer_tool, concept_explainer_tool, cheatsheet_generator_tool]
    agent = create_react_agent(llm, tools, hub.pull("hwchase17/react"))
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True, return_intermediate_steps=True)
    response = executor.invoke({"input": query})
    final_answer = response.get("output", "No answer found.")
    pdf_filename = None
    for _, obs in response.get("intermediate_steps", []):
        if isinstance(obs, str) and obs.startswith("PDF_GENERATED::"):
            try:
                pdf_filename = obs.split("::")[1]
            except IndexError:
                pass
    return final_answer, pdf_filename

# --- 13. Display Chat ---
for entry in st.session_state.chat_history:
    if "user" in entry:
        st.markdown(f"<div class='msg-user'>{entry['user']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='msg-bot'>{entry['bot']}</div>", unsafe_allow_html=True)
        if entry.get("pdf_filename"):
            with open(os.path.join(CHEATSHEET_PATH, entry["pdf_filename"]), "rb") as pdf_file:
                st.download_button("📥 Download Cheatsheet", pdf_file.read(), entry["pdf_filename"], "application/pdf")

# --- 14. Chat Input ---
if user_prompt := st.chat_input("Type your question here..."):
    st.session_state.chat_history.append({"user": user_prompt})
    with st.spinner("Thinking..."):
        answer, pdf_filename = handle_query_logic(user_prompt, st.session_state.get("session_id"))
        st.session_state.chat_history.append({"bot": answer, "pdf_filename": pdf_filename})
    st.rerun()# Place this at the very beginning of your Streamlit script
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="centered")

# --- 1. Device Width Detection ---
if "screen_width" not in st.session_state:
    st.session_state["screen_width"] = 768  # default fallback
    components.html("""
        <script>
        const sendWidth = () => {
            const width = window.innerWidth;
            const streamlitDoc = window.parent.document;
            const data = {"width": width};
            streamlitDoc.dispatchEvent(new CustomEvent("streamlit:screen-width", {detail: data}));
        };
        window.addEventListener("resize", sendWidth);
        sendWidth();
        </script>
    """, height=0, scrolling=False)

# --- 2. Determine Device Type ---
def get_device_type():
    try:
        width = int(st.session_state.get("screen_width", 768))
    except (ValueError, TypeError):
        width = 768
    if width < 576:
        return "mobile"
    elif width < 992:
        return "tablet"
    else:
        return "desktop"

DEVICE = get_device_type()

# --- 3. Theme Variables ---
LIGHT = {"bg": "#f8fafb", "bar": "#fff", "bot": "#e9eef6", "user": "#d1e7dd", "text": "#191b22", "input": "#e8edf2", "border": "#d4dde7", "expander": "#f4f7fb"}
DARK = {"bg": "#18181c", "bar": "#202126", "bot": "#232733", "user": "#22577a", "text": "#f3f5f8", "input": "#242730", "border": "#26282f", "expander": "#24272e"}

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
THEME = DARK if st.session_state["theme"] == "dark" else LIGHT

# --- 4. Responsive CSS Styling ---
msg_font = "0.95rem" if DEVICE == "mobile" else "1.1rem"
padding = "0.8rem" if DEVICE == "mobile" else "1.2rem"
topbar_font = "1.1rem" if DEVICE == "mobile" else "1.5rem"

st.markdown(f"""
<style>
html, body, .stApp {{
    max-width: 100vw;
    overflow-x: hidden;
    padding: 0;
    margin: 0;
    font-family: 'Segoe UI', sans-serif;
    background-color: {THEME['bg']};
    color: {THEME['text']};
}}

.topbar-custom {{
    background-color: {THEME['bar']};
    border-radius: 12px;
    padding: {padding};
    font-size: {topbar_font};
    font-weight: bold;
    text-align: center;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}}

.msg-user, .msg-bot {{
    max-width: 90%;
    padding: {padding};
    margin-bottom: 0.5rem;
    font-size: {msg_font};
    border-radius: 12px;
    border: 1px solid {THEME['border']};
    word-wrap: break-word;
    word-break: break-word;
    box-shadow: 0 1px 5px rgba(0,0,0,0.05);
}}

.msg-user {{
    background-color: {THEME['user']};
    margin-left: auto;
    text-align: right;
    border-radius: 12px 12px 4px 16px;
}}

.msg-bot {{
    background-color: {THEME['bot']};
    margin-right: auto;
    text-align: left;
    border-radius: 12px 12px 16px 4px;
}}

.stButton>button, .stDownloadButton>button {{
    width: 100%;
    padding: 0.6rem 1rem;
    font-size: 1rem;
    border-radius: 10px;
    margin-top: 0.5rem;
    background-color: {THEME['input']};
    border: 1px solid {THEME['border']};
}}

[data-testid="stExpander"] {{
    border: 1px solid {THEME['border']};
    background-color: {THEME['expander']};
    border-radius: 10px;
    font-size: 0.95rem;
}}

.block-container {{
    padding: 0 1rem !important;
}}
</style>
""", unsafe_allow_html=True)

# --- 5. Topbar Display ---
st.markdown(f"<div class='topbar-custom'>Ophthalmology AI Assistant ({DEVICE.title()})</div>", unsafe_allow_html=True)

# --- 6. Theme Switch Buttons ---
col1, col2 = st.columns([10, 1])
with col2:
    if st.button("☀️", key="theme-sun", help="Light mode"):
        st.session_state["theme"] = "light"
        st.rerun()
    if st.button("🌙", key="theme-moon", help="Dark mode"):
        st.session_state["theme"] = "dark"
        st.rerun()

# Now use DEVICE to control layouts and interactions
st.write(f"You are viewing on a **{DEVICE.upper()}** device with screen width: {st.session_state['screen_width']}")
