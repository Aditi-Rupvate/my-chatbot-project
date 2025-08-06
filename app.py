# Place this at the very beginning of your Streamlit script
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
