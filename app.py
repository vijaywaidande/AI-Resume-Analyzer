
import streamlit as st
import re

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AI ATS Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("🚀 AI Resume Analyzer & ATS Scoring")
st.write("Analyze your resume against a job description using AI and NLP.")

# ---------------- LOAD AI MODEL ----------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------- RESUME EXTRACTION ----------------

def extract_resume(uploaded_file):

    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# ---------------- SKILL EXTRACTION ----------------

SKILLS = [
    "Python", "Java", "SQL", "Machine Learning",
    "Deep Learning", "Data Analysis", "Pandas",
    "NumPy", "Scikit-learn", "TensorFlow",
    "React", "JavaScript", "HTML", "CSS",
    "Spring Boot", "AWS", "Docker", "Git",
    "Power BI", "Tableau", "Excel",
    "Data Science", "NLP", "Flask",
    "Django", "MySQL", "MongoDB",
    "PostgreSQL", "Linux", "C++",
    "C", "Communication", "Problem Solving",
    "Statistics", "PyTorch", "REST API",
    "GitHub", "Azure", "Spark"
]


def find_skills(text):

    text = text.lower()

    found = []

    for skill in SKILLS:

        pattern = (
            r"(?<!\w)"
            + re.escape(skill.lower())
            + r"(?!\w)"
        )

        if re.search(pattern, text):
            found.append(skill)

    return found


# ---------------- KEYWORD MATCHING ----------------

STOP_WORDS = {
    "the", "and", "for", "with", "you", "your",
    "are", "our", "will", "this", "that", "from",
    "have", "has", "was", "were", "their", "they",
    "who", "what", "when", "where", "how", "can",
    "all", "any", "not", "but", "into", "about",
    "work", "working", "role", "team", "ability",
    "using", "use", "used", "good", "strong",
    "experience", "required", "preferred"
}


def calculate_keyword_match(resume, jd):

    jd_words = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.]*\b", jd.lower()))

    resume_words = set(
        re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.]*\b", resume.lower())
    )

    important_words = {
        word for word in jd_words
        if len(word) >= 3 and word not in STOP_WORDS
    }

    if not important_words:
        return 0, [], []

    matched = sorted(important_words & resume_words)
    missing = sorted(important_words - resume_words)

    score = (len(matched) / len(important_words)) * 100

    return score, matched, missing


# ---------------- RESUME SECTION CHECK ----------------

def check_resume_sections(text):

    sections = {
        "Education": ["education", "academic qualification"],
        "Skills": ["skills", "technical skills"],
        "Projects": ["projects", "academic projects"],
        "Experience": ["experience", "internship", "work experience"],
        "Certifications": ["certifications", "certificates"]
    }

    text = text.lower()

    found = []
    missing = []

    for section, keywords in sections.items():

        if any(keyword in text for keyword in keywords):
            found.append(section)
        else:
            missing.append(section)

    score = (len(found) / len(sections)) * 100

    return score, found, missing


# ---------------- ATS SCORE CALCULATION ----------------

def calculate_ats_score(
    semantic_score,
    skill_score,
    keyword_score,
    section_score
):

    final_score = (
        semantic_score * 0.35
        + skill_score * 0.35
        + keyword_score * 0.20
        + section_score * 0.10
    )

    return round(final_score, 2)


# ---------------- USER INTERFACE ----------------

st.header("1. Upload Your Resume")

uploaded_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

st.header("2. Enter Job Description")

job_description = st.text_area(
    "Paste the company Job Description",
    height=220
)

# ---------------- ANALYSIS ----------------

if st.button("Analyze Resume", type="primary"):

    if uploaded_file is None:

        st.error("Please upload your resume.")

    elif not job_description.strip():

        st.error("Please enter a job description.")

    else:

        with st.spinner("Analyzing resume using AI..."):

            resume_text = extract_resume(uploaded_file)

            if not resume_text.strip():

                st.error("No readable text found in this PDF.")
                st.stop()

            # AI semantic similarity

            model = load_model()

            embeddings = model.encode([
                resume_text,
                job_description
            ])

            semantic_score = float(
                cosine_similarity(
                    [embeddings[0]],
                    [embeddings[1]]
                )[0][0] * 100
            )

            semantic_score = max(0, min(100, semantic_score))

            # Skill matching

            resume_skills = find_skills(resume_text)
            required_skills = find_skills(job_description)

            matched_skills = [
                skill for skill in required_skills
                if skill in resume_skills
            ]

            missing_skills = [
                skill for skill in required_skills
                if skill not in resume_skills
            ]

            if required_skills:

                skill_score = (
                    len(matched_skills)
                    / len(required_skills)
                ) * 100

            else:

                skill_score = 0

            # Keyword analysis

            keyword_score, matched_keywords, missing_keywords = (
                calculate_keyword_match(
                    resume_text,
                    job_description
                )
            )

            # Resume section analysis

            section_score, found_sections, missing_sections = (
                check_resume_sections(resume_text)
            )

            # Final ATS score

            ats_score = calculate_ats_score(
                semantic_score,
                skill_score,
                keyword_score,
                section_score
            )

        # ---------------- RESULTS DASHBOARD ----------------

        st.success("Resume analysis completed!")

        st.divider()

        st.header("📊 ATS Evaluation Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("ATS Match Score", f"{ats_score}%")
        col2.metric("Semantic Match", f"{semantic_score:.2f}%")
        col3.metric("Skill Match", f"{skill_score:.2f}%")
        col4.metric("Keyword Match", f"{keyword_score:.2f}%")

        st.progress(ats_score / 100)

        if ats_score >= 80:

            st.success("Strong alignment with this job description.")

        elif ats_score >= 60:

            st.warning("Moderate alignment. Some improvements may help.")

        else:

            st.error("Low alignment. Review the missing requirements.")

        st.caption(
            "This is an experimental scoring model, not an official ATS "
            "score or a prediction of interview selection."
        )

        # ---------------- SKILL ANALYSIS ----------------

        st.divider()

        st.header("🎯 Technical Skill Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("✅ Matching Skills")

            if matched_skills:

                for skill in matched_skills:
                    st.write("✔", skill)

            else:

                st.write("No matching skills detected.")

        with col2:

            st.subheader("❌ Missing Skills")

            if missing_skills:

                for skill in missing_skills:
                    st.write("•", skill)

            else:

                st.write("No missing skills detected.")

        # ---------------- KEYWORD ANALYSIS ----------------

        st.divider()

        st.header("🔍 Job Description Keyword Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Matched Keywords")

            st.write(", ".join(matched_keywords[:40]) or "None")

        with col2:

            st.subheader("Missing Keywords")

            st.write(", ".join(missing_keywords[:40]) or "None")

        # ---------------- RESUME SECTIONS ----------------

        st.divider()

        st.header("📄 Resume Section Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Available Sections")

            for section in found_sections:
                st.write("✅", section)

        with col2:

            st.subheader("Sections to Review")

            for section in missing_sections:
                st.write("⚠️", section)

        # ---------------- RECOMMENDATIONS ----------------

        st.divider()

        st.header("💡 AI Resume Improvement Recommendations")

        if missing_skills:

            st.write(
                "Review these job-related skills and include them "
                "only if you genuinely possess them:"
            )

            st.write(", ".join(missing_skills))

        if missing_sections:

            st.write(
                "Consider adding relevant resume sections:",
                ", ".join(missing_sections)
            )

        if missing_keywords:

            st.write(
                "Review the missing JD keywords. Add relevant terms "
                "naturally where your actual experience supports them."
            )

        st.write(
            "Use clear project descriptions, measurable achievements "
            "and consistent formatting."
        )

        # ---------------- DOWNLOAD REPORT ----------------

        report = f"""
AI RESUME ANALYSIS REPORT

ATS MATCH SCORE: {ats_score}%

Semantic Similarity: {semantic_score:.2f}%
Skill Match: {skill_score:.2f}%
Keyword Match: {keyword_score:.2f}%
Resume Completeness: {section_score:.2f}%

MATCHING SKILLS:
{matched_skills}

MISSING SKILLS:
{missing_skills}

MATCHED KEYWORDS:
{matched_keywords}

MISSING KEYWORDS:
{missing_keywords}

AVAILABLE SECTIONS:
{found_sections}

SECTIONS TO REVIEW:
{missing_sections}

Note: This is an experimental resume matching score,
not an official ATS evaluation.
"""

        st.download_button(
            label="📥 Download ATS Report",
            data=report,
            file_name="ATS_Resume_Report.txt",
            mime="text/plain"
        )
