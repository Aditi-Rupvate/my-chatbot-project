import streamlit as st
import requests
import speech_recognition as sr

# --- THIS IS THE URL YOU WILL GET FROM RENDER AFTER DEPLOYING YOUR BACKEND ---
BACKEND_URL = "https://your-render-app-name.onrender.com"  # <--- REPLACE THIS

# --- Theme colors and Session State Initialization ---
LIGHT = {
    "bg": "#f8fafb", "bar": "#fff", "bot": "#e9eef6", "user": "#d1e7dd",
    "text": "#191b22", "input": "#e8edf2", "border": "#d4dde7", "expander": "#f4f7fb"
}
DARK = {
    "bg": "#18181c", "bar": "#202126", "bot": "#232733", "user": "#22577a",
    "text": "#f3f5f8", "input": "#242730", "border": "#26282f", "expander": "#24272e"
}

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
THEME = DARK if st.session_state["theme"] == "dark" else LIGHT

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None
if "active_doc_name" not in st.session_state:
    st.session_state["active_doc_name"] = None

st.set_page_config(layout="centered")

# --- UI Styling ---
st.markdown(
    f"""
    <style>
    html, body, .stApp {{ background: {THEME['bg']} !important; color: {THEME['text']} !important; }}
    .topbar-custom {{ background: {THEME['bar']}; border-radius: 0 0 16px 16px; padding: 1.3em 1.2em 1.15em 2.1em; margin-bottom: 1.6em; box-shadow: 0 2px 12px 0 rgba(44,46,66,0.06); display: flex; align-items: center; justify-content: space-between; font-size:1.55rem; font-weight: 800; letter-spacing:.02em; }}
    .msg-user {{ background: {THEME['user']}; color: {THEME['text']}; border-radius: 16px 16px 4px 20px; margin-bottom: 0.3em; padding: 1em 1.35em; width: fit-content; max-width: 85%; font-size: 1.13rem; border: 1.5px solid {THEME['border']}; margin-left: auto; margin-right: 0; text-align: right; box-shadow: 0 1px 12px 0 rgba(55,96,148,0.05); word-break: break-word; }}
    .msg-bot {{ background: {THEME['bot']}; color: {THEME['text']}; border-radius: 16px 16px 20px 4px; margin-bottom: 0.7em; padding: 1.08em 1.23em 1em 1.18em; width: fit-content; max-width: 85%; font-size: 1.13rem; border: 1.5px solid {THEME['border']}; margin-right: auto; margin-left: 0; text-align: left; box-shadow: 0 1px 12px 0 rgba(44,46,66,0.05); word-break: break-word; }}
    </style>
    """, unsafe_allow_html=True
)

# --- Top bar ---
col1, col2 = st.columns([8,1])
with col1:
    st.markdown("<div class='topbar-custom'>Ophthalmology AI Assistant</div>", unsafe_allow_html=True)
with col2:
    if st.button("☀️", key="theme-sun"): st.session_state["theme"] = "light"; st.rerun()
    if st.button("🌙", key="theme-moon"): st.session_state["theme"] = "dark"; st.rerun()

# --- Chat History ---
for entry in st.session_state.chat_history:
    if "user" in entry:
        st.markdown(f"<div class='msg-user'>{entry['user']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='msg-bot'>{entry['bot']}</div>", unsafe_allow_html=True)
        if entry.get("pdf_url"):
            pdf_response = requests.get(entry["pdf_url"])
            st.download_button("📥 Download Cheatsheet", pdf_response.content, "cheatsheet.pdf", "application/pdf")

# --- Document Upload Expander ---
with st.expander("Upload a Custom Document"):
    uploaded_file = st.file_uploader("Upload a PDF to ask questions about it", type="pdf")
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=300)
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state["session_id"] = data["session_id"]
                    st.session_state["active_doc_name"] = uploaded_file.name
                    st.session_state.chat_history.append({"bot": f"Ready for questions about **{uploaded_file.name}**."})
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to process document: {e}")

# --- Active Document Status ---
if st.session_state["active_doc_name"]:
    st.info(f"Active Document: **{st.session_state['active_doc_name']}**")
    if st.button("Clear Document & Revert to Default"):
        st.session_state["session_id"] = None
        st.session_state["active_doc_name"] = None
        st.session_state.chat_history.append({"bot": "Reverted to default knowledge base."})
        st.rerun()

# --- Input Handling Section ---
final_prompt = None
st.markdown("---")
if st.button("🎤 Speak your question"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.toast("Listening...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=15)
            st.toast("Transcribing...")
            final_prompt = r.recognize_google(audio)
        except Exception as e:
            st.error(f"Voice input error: {e}")

text_prompt = st.chat_input("Type your question here...")
if text_prompt:
    final_prompt = text_prompt

# --- API Call Logic ---
if final_prompt:
    st.session_state.chat_history.append({"user": final_prompt})
    with st.spinner("Thinking..."):
        try:
            payload = { "query": final_prompt, "session_id": st.session_state.get("session_id") }
            resp = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer", "I'm sorry, I couldn't find an answer.")
            pdf_path = data.get("pdf_url")
            full_pdf_url = f"{BACKEND_URL}{pdf_path}" if pdf_path else None
            
            st.session_state.chat_history.append({"bot": answer, "pdf_url": full_pdf_url})
        except Exception as e:
            st.session_state.chat_history.append({"bot": f"**API Error:** {e}", "pdf_url": None})
    st.rerun()
