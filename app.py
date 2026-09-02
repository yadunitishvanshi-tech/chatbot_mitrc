"""
MITRC College Alwar - Next-Gen AI Portal & Smart Student Hub
============================================================
Fixes:
- Pure High-Contrast Dark UI with Crisp White Text
- Fixed Chat Input & Text Input Visibility (No Blackout Text)
- Clean Glassmorphism Cards & Pulsing Indicators

Run command: streamlit run app.py
"""

import io
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangChain & Hugging Face Imports
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# =========================================================
# 1. LANGCHAIN PROMPT TEMPLATES
# =========================================================

SYSTEM_PROMPT_TEXT = """
You are "MITRC AI", the official virtual academic assistant and student guide for MITRC College (Modern Institute of Technology and Research Centre), located in Alwar, Rajasthan.

Your Core Directives:
1. REPRESENTATION: Always maintain a polite, respectful, clear, and encouraging tone as an official representative of MITRC Alwar.
2. ACADEMIC & LEARNING HELP:
   - When students ask study or subject questions (e.g., Programming in Python/C++/Java, DBMS, Microprocessors, Mathematics, Engineering concepts), explain clearly step-by-step.
   - Provide neat code snippets, examples, or structured points whenever relevant to aid learning for university exam preparation.
3. CAMPUS FAQ GUIDANCE:
   - Use the provided FAQ Context to answer campus-related questions accurately (Admissions, Fees, Courses, Hostel, Library, Placements, Exams).
   - If information is not in the context, guide the student to visit the MITRC admin office or contact standard support channels politely.
4. BOUNDARIES: Focus exclusively on academic guidance, learning assistance, and college services.

Context Information (Campus Knowledge):
{faq_context}

User Query:
{user_query}

Your Response (as MITRC AI Assistant):
"""

COLLEGE_SYSTEM_PROMPT = PromptTemplate(
    input_variables=["faq_context", "user_query"],
    template=SYSTEM_PROMPT_TEXT
)

RECEIPT_PROMPT_TEXT = """
An official visual fee payment document receipt for MITRC College Alwar (Modern Institute of Technology and Research Centre).
Design: Elegant maroon header banner with gold text "MITRC COLLEGE ALWAR - PAYMENT RECEIPT".
Information structured on crisp white canvas:
- Student ID: {student_id}
- Student Name: {student_name}
- Academic Program: {program}
- Fee Amount / Status: {amount} ({status})
Details: Includes official MITRC seal watermark, verified stamp, bar code at footer, crisp typography, clean paper texture, studio quality lighting.
"""

RECEIPT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["student_id", "student_name", "program", "amount", "status"],
    template=RECEIPT_PROMPT_TEXT
)


# =========================================================
# 2. PATHS & DATABASE SETUP
# =========================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_DIR.mkdir(exist_ok=True)

DB_PATH = DATABASE_DIR / "college.db"
CSV_PATH = DATA_DIR / "students_demo.csv"


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_database():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT,
            program TEXT,
            degree TEXT,
            semester INTEGER,
            year INTEGER,
            section TEXT,
            email TEXT,
            phone TEXT,
            date_of_birth TEXT,
            attendance_pct REAL,
            cgpa REAL,
            fee_status TEXT,
            fee_due INTEGER,
            library_books INTEGER,
            hostel TEXT
        )
    """)
    conn.commit()
    conn.close()


def import_students():
    if not CSV_PATH.exists():
        return

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM students")
    count = cursor.fetchone()[0]

    if count == 0:
        df = pd.read_csv(CSV_PATH)
        df.to_sql("students", conn, if_exists="append", index=False)

    conn.close()


create_database()
import_students()


# =========================================================
# 3. AI MODELS CACHING
# =========================================================

@st.cache_resource
def load_chat_model():
    if not HF_TOKEN:
        return None
    try:
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            provider="featherless-ai",
            huggingfacehub_api_token=HF_TOKEN,
            max_new_tokens=500,
            temperature=0.6,
        )
        return ChatHuggingFace(llm=llm)
    except Exception:
        return None


@st.cache_resource
def load_image_client():
    if not HF_TOKEN:
        return None
    try:
        return InferenceClient(provider="fal-ai", api_key=HF_TOKEN)
    except Exception:
        return None


chat_model = load_chat_model()
image_client = load_image_client()


# =========================================================
# 4. CAMPUS FAQ KNOWLEDGE BASE
# =========================================================

FAQ_KB = {
    "admissions": "MITRC Admissions: 10+2 with PCM (min 60%) for B.Tech. Apply online via college portal with ₹1,000 application fee.",
    "fees": "MITRC Tuition Fees: ₹80,000–₹1,50,000/year. Payments accepted via NetBanking, UPI, or DD in 2 installments.",
    "courses": "MITRC Programs offered: B.Tech and M.Tech in Computer Science (CSE), AI & ML, Data Science, ECE, Mechanical, and Civil Engineering.",
    "hostel": "MITRC Campus Hostel: Modern separate hostel facilities for boys and girls with AC/Non-AC rooms and 24x7 mess facility.",
    "library": "MITRC Central Library: Open 8 AM to 8 PM on weekdays. Students can issue up to 4 books for 14 days.",
    "placements": "MITRC Training & Placement Cell: Dedicated placement drives (Aug–Mar). Key recruiters include TCS, Infosys, Wipro, and Capgemini.",
    "exams": "MITRC Exams: Semester mid-terms and end-term university examinations conducted as per Bikaner Technical University (BTU) schedule.",
    "contact": "MITRC Contact Info: Email info@mitrc.ac.in | Helpline +91 144 2881000 | Office hours 9 AM – 5 PM (Mon-Sat)."
}


def get_faq_context(question: str) -> str:
    q_words = set(re.findall(r"[a-zA-Z]+", question.lower()))
    matches = [text for topic, text in FAQ_KB.items() if any(w in topic for w in q_words)]
    return "\n".join(matches) if matches else "General MITRC College guidelines apply."


# =========================================================
# 5. STREAMLIT UI CONFIG & STRICT VISIBILITY CSS
# =========================================================

st.set_page_config(
    page_title="MITRC Alwar - Smart AI Campus Portal",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
/* 1. Global Dark Background */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0b0f19 !important;
    color: #f3f4f6 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* 2. Text Visibility Fixes - Universal Rule */
p, span, label, h1, h2, h3, h4, h5, h6, li {
    color: #f3f4f6 !important;
}

/* 3. Text Inputs & Chat Inputs Fix (No Black-on-Black) */
input, textarea, [data-baseweb="input"] input {
    color: #ffffff !important;
    background-color: #1e293b !important;
    border: 1px solid #dc2626 !important;
    border-radius: 8px !important;
}

div[data-baseweb="base-input"] {
    background-color: #1e293b !important;
}

/* Streamlit Chat Input Box Fix */
.stChatInputContainer textarea {
    color: #ffffff !important;
    background-color: #1e293b !important;
    border: 1px solid #ef4444 !important;
}

.stChatInputContainer textarea::placeholder {
    color: #9ca3af !important;
}

/* Chat Message Bubbles */
[data-testid="stChatMessage"] {
    background-color: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    padding: 12px !important;
    margin-bottom: 10px !important;
}

/* 4. Pulsing Animations */
@keyframes pulseRed {
    0% { box-shadow: 0 0 10px rgba(220, 38, 38, 0.4); }
    50% { box-shadow: 0 0 25px rgba(239, 68, 68, 0.8); }
    100% { box-shadow: 0 0 10px rgba(220, 38, 38, 0.4); }
}

@keyframes blinkIndicator {
    0% { opacity: 1; }
    50% { opacity: 0.3; }
    100% { opacity: 1; }
}

/* Header Banner */
.mitrc-header {
    background: linear-gradient(135deg, #310404 0%, #7f1d1d 100%);
    border: 2px solid #ef4444;
    animation: pulseRed 3s infinite ease-in-out;
    padding: 20px 28px;
    border-radius: 14px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.mitrc-title {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff !important;
    margin: 0;
}

.mitrc-subtitle {
    color: #4ade80 !important;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Status Indicator */
.status-pill {
    display: inline-flex;
    align-items: center;
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid #22c55e;
    color: #4ade80 !important;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}

.blink-dot {
    height: 9px;
    width: 9px;
    background-color: #22c55e;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    animation: blinkIndicator 1.5s infinite ease-in-out;
    box-shadow: 0 0 8px #22c55e;
}

/* Cards & Glassmorphism */
.mitrc-card {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-top: 3px solid #dc2626;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

/* KPI Scorecards */
.kpi-card {
    background-color: #1f2937;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}

.kpi-value {
    font-size: 24px;
    font-weight: 700;
    color: #38bdf8 !important;
    margin-top: 4px;
}

.kpi-label {
    font-size: 11px;
    color: #9ca3af !important;
    letter-spacing: 0.8px;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #0d111a !important;
    border-right: 1px solid #1f2937 !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 6. HEADER & SIDEBAR BRANDING
# =========================================================

with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding:10px 0;">
            <h2 style="color:#ef4444 !important; margin:0; font-weight:800;">MITRC ALWAR</h2>
            <p style="color:#4ade80 !important; font-size:12px; margin-top:2px; font-weight:600;">Smart Campus Assistant</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### ⚡ Quick Nav")
    st.markdown("• 💬 **AI Study & Campus Bot**")
    st.markdown("• 📊 **Student Self-Portal**")
    st.markdown("• 🧾 **AI Fee Receipt Hub**")
    st.markdown("• 🗄️ **Live Database Console**")
    
    st.markdown("---")
    if DB_PATH.exists():
        conn = get_connection()
        total_s = pd.read_sql_query("SELECT COUNT(*) as c FROM students", conn).iloc[0]["c"]
        conn.close()
        st.markdown(f"""
            <div class="status-pill" style="width: 100%; justify-content: center;">
                <span class="blink-dot"></span> <b>System Live:</b> {total_s:,} Students
            </div>
        """, unsafe_allow_html=True)

# Main Banner Header
st.markdown("""
    <div class="mitrc-header">
        <div>
            <div class="mitrc-title">🎓 MITRC ALWAR AI PORTAL</div>
            <div class="mitrc-subtitle">Modern Institute of Technology & Research Centre</div>
        </div>
        <div>
            <span class="status-pill">
                <span class="blink-dot"></span> ONLINE
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs([
    "💬 AI Academic & Campus Assistant", 
    "📊 Student Portal & Fee Receipt", 
    "🗂️ Live Database Manager"
])


# =========================================================
# TAB 1: NEXT-GEN AI CHATBOT INTERFACE
# =========================================================

with tab1:
    st.markdown('<h3>🤖 Ask MITRC Academic AI</h3>', unsafe_allow_html=True)
    st.caption("Ask questions regarding your subjects (Python, Java, DBMS), BTU exam syllabus, fees, or hostel rules.")

    # Quick prompt chip suggestions
    st.markdown("**Quick Topics:**")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    if q_col1.button("📚 B.Tech Syllabus", use_container_width=True):
        st.session_state.quick_q = "What is the syllabus structure for B.Tech CSE?"
    if q_col2.button("💰 Fee Payment Rules", use_container_width=True):
        st.session_state.quick_q = "Explain the college tuition fee structure and payment modes."
    if q_col3.button("💻 Python Code Help", use_container_width=True):
        st.session_state.quick_q = "Explain Python lists vs tuples with code examples."
    if q_col4.button("🏢 Hostel Rules", use_container_width=True):
        st.session_state.quick_q = "What are the hostel room facilities and timing rules?"

    st.markdown("<br>", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        avatar = "👤" if msg["role"] == "user" else "🎓"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # User Input Handling
    default_input = st.session_state.pop("quick_q", "")
    user_q = st.chat_input("Type your academic or college question here...", key="chat_input")
    
    if default_input and not user_q:
        user_q = default_input

    if user_q:
        st.chat_message("user", avatar="👤").markdown(user_q)
        st.session_state.chat_history.append({"role": "user", "content": user_q})

        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("MITRC AI is generating your response..."):
                if chat_model:
                    faq_ctx = get_faq_context(user_q)
                    prompt = COLLEGE_SYSTEM_PROMPT.format(faq_context=faq_ctx, user_query=user_q)
                    response = chat_model.invoke(prompt).content
                else:
                    response = "⚠️ System Notice: Chat model is initializing or HF_TOKEN is missing in `.env`."

                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})


# =========================================================
# TAB 2: DYNAMIC STUDENT PORTAL & REAL-LOOKING RECEIPT
# =========================================================

with tab2:
    st.markdown('<h3>🔒 Student Portal Verification</h3>', unsafe_allow_html=True)
    st.caption("Secure login using your Student ID and Date of Birth to view live records.")

    v1, v2 = st.columns(2)
    s_id = v1.text_input("Student ID", placeholder="e.g. MITRC00001", key="input_sid")
    s_dob = v2.date_input("Date of Birth", value=None, min_value=pd.Timestamp("1995-01-01"), key="input_dob")

    verify = st.button("🔓 Access Record", use_container_width=True)

    if verify and s_id and s_dob:
        conn = get_connection()
        df = pd.read_sql_query("SELECT * FROM students WHERE student_id = ?", conn, params=(s_id.upper().strip(),))
        conn.close()

        if df.empty:
            st.error("❌ Invalid Record: Student ID not found.")
        elif str(df.iloc[0]["date_of_birth"]) != s_dob.isoformat():
            st.error("❌ Authentication Failed: Student ID and Date of Birth do not match.")
        else:
            s = df.iloc[0]
            st.toast(f"Authenticated successfully for {s['name']}!", icon="✅")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="mitrc-card" style="border-top:3px solid #22c55e;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <h3 style="margin:0; color:#4ade80 !important;">Welcome, {s['name']} 👋</h3>
                        <span style="background:rgba(34,197,94,0.2); color:#4ade80 !important; border:1px solid #22c55e; padding:4px 12px; border-radius:20px; font-weight:bold;">
                            {s['fee_status'].upper()}
                        </span>
                    </div>
                    <p style="color:#9ca3af !important; font-size:14px;">
                        <b>Program:</b> {s['program']} ({s['degree']}) &nbsp;|&nbsp; 
                        <b>Semester:</b> {s['semester']} &nbsp;|&nbsp; 
                        <b>Section:</b> {s['section']} &nbsp;|&nbsp; 
                        <b>Email:</b> {s['email']}
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Dynamic KPI Grid
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            with kpi_col1:
                st.markdown(f"""
                    <div class="kpi-card" style="border-left-color: {'#22c55e' if s['attendance_pct']>=75 else '#ef4444'};">
                        <div class="kpi-label">ATTENDANCE</div>
                        <div class="kpi-value" style="color: {'#4ade80' if s['attendance_pct']>=75 else '#f87171'} !important;">
                            {s['attendance_pct']}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with kpi_col2:
                st.markdown(f"""
                    <div class="kpi-card" style="border-left-color: #38bdf8;">
                        <div class="kpi-label">CURRENT CGPA</div>
                        <div class="kpi-value">{s['cgpa']}</div>
                    </div>
                """, unsafe_allow_html=True)

            with kpi_col3:
                st.markdown(f"""
                    <div class="kpi-card" style="border-left-color: #ef4444;">
                        <div class="kpi-label">FEE DUE</div>
                        <div class="kpi-value" style="color:#f87171 !important;">₹{int(s['fee_due']):,}</div>
                    </div>
                """, unsafe_allow_html=True)

            with kpi_col4:
                st.markdown(f"""
                    <div class="kpi-card" style="border-left-color: #a855f7;">
                        <div class="kpi-label">LIBRARY BOOKS</div>
                        <div class="kpi-value" style="color:#c084fc !important;">{int(s['library_books'])} Issued</div>
                    </div>
                """, unsafe_allow_html=True)

            # Attendance Progress Bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(
                min(int(s['attendance_pct']), 100) / 100, 
                text=f"Attendance Target (Min 75% Required) - Current: {s['attendance_pct']}%"
            )

            # AI Fee Receipt Generator
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<h3>🧾 AI Official Fee Receipt Generator</h3>', unsafe_allow_html=True)

            if st.button("✨ Generate AI Fee Receipt Image", use_container_width=True):
                with st.spinner("FLUX AI is crafting your official receipt..."):
                    if image_client:
                        try:
                            prompt = RECEIPT_PROMPT_TEMPLATE.format(
                                student_id=s['student_id'],
                                student_name=s['name'],
                                program=s['program'],
                                amount=f"₹{s['fee_due']}",
                                status=s['fee_status']
                            )
                            img = image_client.text_to_image(
                                prompt=prompt,
                                model="black-forest-labs/FLUX.1-schnell",
                                width=768,
                                height=512,
                                num_inference_steps=4
                            )

                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            img_bytes = buf.getvalue()

                            st.image(img, caption=f"Generated Receipt for {s['student_id']}", use_container_width=True)
                            
                            st.download_button(
                                "⬇️ Download Official Fee Receipt Image",
                                data=img_bytes,
                                file_name=f"MITRC_Receipt_{s['student_id']}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Receipt Generation Error: {e}")
                    else:
                        st.error("Hugging Face Engine missing token. Please ensure HF_TOKEN is in .env.")


# =========================================================
# TAB 3: ADMIN DATA MANAGEMENT CONSOLE
# =========================================================

with tab3:
    st.markdown('<h3>🗄️ Database Console & Source Files</h3>', unsafe_allow_html=True)

    dt1, dt2 = st.tabs(["📄 Source CSV Data", "🗄️ Live SQLite Database"])

    with dt1:
        if CSV_PATH.exists():
            csv_df = pd.read_csv(CSV_PATH)
            st.caption(f"File Path: `{CSV_PATH}` — Total Records: {len(csv_df)}")
            st.dataframe(csv_df, use_container_width=True, height=350)
        else:
            st.warning("No CSV file found at `data/students_demo.csv`.")

    with dt2:
        if DB_PATH.exists():
            conn = get_connection()
            db_df = pd.read_sql_query("SELECT * FROM students", conn)
            conn.close()
            st.caption(f"Database Path: `{DB_PATH}` — Table: `students` — Total Rows: {len(db_df)}")

            search = st.text_input("🔍 Search Student Records", placeholder="Search by name or ID...", key="db_search")
            if search.strip():
                mask = (
                    db_df["student_id"].astype(str).str.contains(search, case=False, na=False) |
                    db_df["name"].astype(str).str.contains(search, case=False, na=False)
                )
                db_df = db_df[mask]

            st.dataframe(db_df, use_container_width=True, height=350)