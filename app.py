
import re
import io
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="HireSense AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 HireSense AI")
st.subheader("AI Recruitment & Skill Intelligence Platform")
st.write(
    "Analyze resumes, match candidates with job descriptions, "
    "and discover skill gaps using NLP and machine learning."
)

SKILL_TAXONOMY = {
    "Python": ["python"],
    "Java": ["java"],
    "SQL": ["sql", "mysql", "postgresql"],
    "Machine Learning": ["machine learning", "scikit-learn"],
    "Deep Learning": ["deep learning", "pytorch", "tensorflow"],
    "NLP": ["nlp", "natural language processing", "spacy"],
    "LLMs": ["llm", "llms", "large language model", "transformers"],
    "RAG": ["rag", "retrieval augmented generation", "vector database"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Data Visualization": ["matplotlib", "seaborn", "power bi", "tableau"],
    "FastAPI": ["fastapi"],
    "Flask": ["flask"],
    "Git": ["git", "github"],
    "Docker": ["docker"],
    "AWS": ["aws", "amazon web services"],
    "REST APIs": ["rest api", "rest apis"],
    "Excel": ["excel"],
    "JavaScript": ["javascript"],
    "HTML/CSS": ["html", "css"]
}


def extract_skills(text):
    text = str(text).lower()
    found = []

    for skill, aliases in SKILL_TAXONOMY.items():
        for alias in aliases:
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"
            if re.search(pattern, text):
                found.append(skill)
                break

    return sorted(set(found))


def calculate_match(resume_text, job_text):
    documents = [str(resume_text), str(job_text)]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))

    if job_skills:
        skill_coverage = len(resume_skills & job_skills) / len(job_skills)
    else:
        skill_coverage = 0.0

    score = (0.60 * similarity + 0.40 * skill_coverage) * 100

    return round(score, 2), sorted(resume_skills), sorted(job_skills)


tab1, tab2, tab3 = st.tabs([
    "Resume Matcher",
    "Dataset Analysis",
    "About"
])

with tab1:
    st.header("Match a Resume to a Job")

    resume_file = st.file_uploader(
        "Upload a resume (.txt)",
        type=["txt"],
        key="resume"
    )

    resume_text_input = st.text_area(
        "Or paste the resume text here",
        height=180,
        placeholder="Paste candidate skills, education, projects, and experience..."
    )

    job_text = st.text_area(
        "Paste the job description",
        height=180,
        placeholder="Enter the role, responsibilities, and required skills..."
    )

    if st.button("Analyze Resume", type="primary"):
        if not (resume_file or resume_text_input.strip()):
            st.error("Please upload a resume or paste resume text.")
        elif not job_text.strip():
            st.error("Please enter a job description.")
        else:
            if resume_file:
                uploaded_text = resume_file.getvalue().decode(
                    "utf-8", errors="replace"
                )
                resume_text = uploaded_text or resume_text_input
            else:
                resume_text = resume_text_input

            score, resume_skills, job_skills = calculate_match(
                resume_text, job_text
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Experimental Match Score", f"{score}%")

            with col2:
                st.metric(
                    "Required Skills Detected",
                    f"{len(set(resume_skills) & set(job_skills))}/{len(job_skills)}"
                )

            st.progress(min(max(score / 100, 0.0), 1.0))

            st.subheader("Skills Found in Resume")
            st.write(", ".join(resume_skills) if resume_skills else "No listed skills detected.")

            st.subheader("Skills Detected in Job Description")
            st.write(", ".join(job_skills) if job_skills else "No listed skills detected.")

            missing = sorted(set(job_skills) - set(resume_skills))

            st.subheader("Potential Skill Gaps")
            if missing:
                st.warning("Skills not detected in this resume: " + ", ".join(missing))
            else:
                st.success("All detected job skills were also detected in the resume.")

            st.caption(
                "This score is an experimental text-similarity and skill-coverage "
                "measure, not a probability of job success. Verify results manually."
            )

with tab2:
    st.header("Analyze Your Synthetic Dataset")

    dataset_file = st.file_uploader(
        "Upload hiresense_500_synthetic_resumes.csv",
        type=["csv"],
        key="dataset"
    )

    if dataset_file:
        try:
            df = pd.read_csv(dataset_file)
            st.success(f"Loaded {len(df)} records.")

            st.subheader("Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)

            st.subheader("Dataset Summary")
            st.write(df.describe(include="all").transpose())

            if "target_role" in df.columns:
                st.subheader("Candidates by Target Role")
                st.bar_chart(df["target_role"].value_counts())

            if "experience_years" in df.columns:
                st.subheader("Experience Distribution")
                st.bar_chart(
                    df["experience_years"].value_counts().sort_index()
                )

        except Exception as error:
            st.error(f"Could not read the CSV: {error}")
    else:
        st.info("Upload the synthetic resumes CSV to explore the dataset.")

with tab3:
    st.header("About HireSense AI")

    st.markdown("""
    **Current capabilities**
    - NLP-based skill extraction
    - TF-IDF resume/job similarity
    - Skill coverage analysis
    - Potential skill-gap identification
    - Synthetic dataset exploration

    **Planned extensions**
    - Sentence Transformer semantic search
    - Retrieval-Augmented Generation (RAG)
    - LLM-based explanations
    - Resume PDF parsing
    - Candidate comparison dashboard
    """)

    st.warning(
        "Educational prototype only. This application uses experimental "
        "matching scores and must not be used for automated hiring decisions."
    )
