import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

load_dotenv()

MODELS = {
    "OpenAI GPT-OSS 120B (best quality)": "openai/gpt-oss-120b",
    "OpenAI GPT-OSS 20B (fastest)": "openai/gpt-oss-20b",
}


def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


def stream_reply(client, model, system, user, temperature):
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        d = chunk.choices[0].delta.content
        if d:
            yield d


def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


CAREER_SYSTEM = (
    "You are an expert career counsellor for students and freshers in India. "
    "Give practical, honest, personalised guidance. Use markdown with clear headings."
)

RESUME_SYSTEM = (
    "You are an expert resume reviewer and ATS specialist. "
    "Be specific and honest. Never invent experience, skills or numbers that the "
    "candidate has not provided. Use markdown with clear headings."
)

st.set_page_config(page_title="AI Career Guidance & Resume Optimizer", page_icon="🎯", layout="wide")
st.title("🎯 AI Career Guidance & Resume Optimizer")
st.caption("Get a personalised career roadmap and make your resume ATS-friendly.")

with st.sidebar:
    st.header("Settings")
    api_key = get_api_key() or st.text_input("Groq API key", type="password")
    model_label = st.selectbox("Model", list(MODELS.keys()))
    temperature = st.slider("Creativity", 0.0, 1.0, 0.4, 0.1)

tab1, tab2 = st.tabs(["🧭 Career Guidance", "📄 Resume Optimizer"])

# ---------------- Career Guidance ----------------
with tab1:
    st.subheader("Tell us about yourself")
    c1, c2 = st.columns(2)
    with c1:
        education = st.text_input("Education", placeholder="e.g. BCA 3rd year")
        skills = st.text_area("Your skills", placeholder="e.g. Python, SQL, HTML, communication")
    with c2:
        interests = st.text_area("Interests", placeholder="e.g. AI, web development, data")
        goal = st.text_input("Career goal (optional)", placeholder="e.g. Data Analyst, Software Developer")

    if st.button("Get Career Guidance", type="primary"):
        if not api_key:
            st.error("Please add your Groq API key in the sidebar.")
        elif not (education.strip() and skills.strip()):
            st.warning("Please fill at least Education and Skills.")
        else:
            client = Groq(api_key=api_key)
            prompt = f"""Student profile:
- Education: {education}
- Skills: {skills}
- Interests: {interests or 'Not specified'}
- Career goal: {goal or 'Not decided'}

Give me:
1. Top 3-5 suitable career paths with a short reason for each
2. Skill gap: what I already have vs what I must learn
3. A 6-month step-by-step roadmap (month-wise)
4. Free/low-cost learning resources and 3 project ideas
5. Entry-level job titles to search for and the expected salary range in India (mention it is approximate)
"""
            try:
                with st.spinner("Preparing your roadmap..."):
                    st.write_stream(stream_reply(client, MODELS[model_label], CAREER_SYSTEM, prompt, temperature))
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# ---------------- Resume Optimizer ----------------
with tab2:
    st.subheader("Upload or paste your resume")
    uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"])
    pasted = st.text_area("...or paste resume text", height=200)
    job_desc = st.text_area(
        "Target job description (optional but recommended)",
        height=150,
        placeholder="Paste the job posting here for a better match score.",
    )

    if st.button("Optimize Resume", type="primary"):
        resume_text = pasted.strip()
        if uploaded is not None:
            try:
                resume_text = read_pdf(uploaded) or resume_text
            except Exception as e:
                st.error(f"Could not read the PDF: {e}")

        if not api_key:
            st.error("Please add your Groq API key in the sidebar.")
        elif len(resume_text) < 100:
            st.warning("Please upload a PDF or paste your resume text (the PDF may be scanned/image-only).")
        else:
            client = Groq(api_key=api_key)
            prompt = f"""RESUME:
{resume_text[:12000]}

TARGET JOB DESCRIPTION:
{job_desc[:6000] if job_desc.strip() else 'Not provided - review for general fresher/entry-level roles.'}

Give me:
1. ATS score out of 100 with a short explanation
2. Strengths
3. Weaknesses and missing sections
4. Missing keywords (compare with the job description if given)
5. Rewrite 5 weak bullet points stronger using action verbs (only use facts present in my resume; use [add number] where a metric is missing)
6. A improved professional summary (3 lines)
7. Final checklist of fixes, in priority order
"""
            try:
                with st.spinner("Analysing your resume..."):
                    st.write_stream(stream_reply(client, MODELS[model_label], RESUME_SYSTEM, prompt, temperature))
            except Exception as e:
                st.error(f"Something went wrong: {e}")
