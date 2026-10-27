
import re
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="HireSense AI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 HireSense AI")
st.subheader("AI Recruitment & Skill Intelligence Platform")
st.write(
    "Compare resumes with job descriptions using semantic AI, "
    "NLP, and skill-gap analysis."
)

# ---------------- SKILL TAXONOMY ----------------

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


# ---------------- AI MODEL ----------------

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def calculate_tfidf(resume_text, job_text):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(
        [str(resume_text), str(job_text)]
    )

    similarity = cosine_similarity(
        matrix[0:1], matrix[1:2]
    )[0][0]

    return float(similarity * 100)


def calculate_semantic_similarity(resume_text, job_text):
    model = load_semantic_model()

    embeddings = model.encode(
        [str(resume_text), str(job_text)],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    similarity = float(embeddings[0] @ embeddings[1])

    # Keep the displayed percentage between 0 and 100.
    return max(0.0, min(100.0, similarity * 100))


def analyze_resume(resume_text, job_text):
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))

    if job_skills:
        coverage = (
            len(resume_skills.intersection(job_skills))
            / len(job_skills)
        )
    else:
        coverage = 0.0

    tfidf_score = calculate_tfidf(resume_text, job_text)
    semantic_score = calculate_semantic_similarity(
        resume_text, job_text
    )

    # Experimental combined score:
    # 60% semantic similarity + 40% detected skill coverage.
    combined_score = (
        0.60 * semantic_score + 0.40 * coverage * 100
    )

    return {
        "semantic_score": round(semantic_score, 2),
        "tfidf_score": round(tfidf_score, 2),
        "coverage": round(coverage * 100, 2),
        "combined_score": round(combined_score, 2),
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "missing_skills": sorted(job_skills - resume_skills),
        "matched_skills": sorted(job_skills & resume_skills)
    }


# ---------------- APP TABS ----------------

tab1, tab2, tab3 = st.tabs([
    "Resume Matcher",
    "Dataset Analysis",
    "About"
])


# ---------------- RESUME MATCHER ----------------

with tab1:
    st.header("Semantic AI Resume Matcher")

    st.info(
        "The first semantic analysis may take a few minutes "
        "because the AI model needs to download and initialize."
    )

    resume_file = st.file_uploader(
        "Upload a resume (.txt)",
        type=["txt"],
        key="resume_upload"
    )

    resume_input = st.text_area(
        "Or paste resume text here",
        height=180,
        placeholder="Enter skills, education, projects, and experience..."
    )

    job_text = st.text_area(
        "Paste the job description",
        height=180,
        placeholder="Enter the role, responsibilities, and required skills..."
    )

    if st.button("Run Semantic AI Analysis", type="primary"):
        if resume_file is not None:
            uploaded_text = resume_file.getvalue().decode(
                "utf-8", errors="replace"
            )
            resume_text = uploaded_text or resume_input
        else:
            resume_text = resume_input

        if not resume_text.strip():
            st.error("Please upload a resume or paste resume text.")
        elif not job_text.strip():
            st.error("Please enter a job description.")
        else:
            try:
                with st.spinner(
                    "Loading semantic AI model and analyzing text..."
                ):
                    results = analyze_resume(resume_text, job_text)

                st.success("Analysis completed!")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Combined Experimental Score",
                    f"{results['combined_score']}%"
                )

                col2.metric(
                    "Semantic Similarity",
                    f"{results['semantic_score']}%"
                )

                col3.metric(
                    "Skill Coverage",
                    f"{results['coverage']}%"
                )

                st.progress(
                    max(0.0, min(
                        1.0, results["combined_score"] / 100
                    ))
                )

                st.subheader("Baseline vs Semantic AI")

                compare_df = pd.DataFrame({
                    "Method": [
                        "TF-IDF",
                        "Semantic AI",
                        "Combined Score"
                    ],
                    "Score (%)": [
                        results["tfidf_score"],
                        results["semantic_score"],
                        results["combined_score"]
                    ]
                })

                st.bar_chart(
                    compare_df.set_index("Method")
                )

                st.dataframe(
                    compare_df,
                    hide_index=True,
                    use_container_width=True
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Skills Found in Resume")
                    st.write(
                        ", ".join(results["resume_skills"])
                        if results["resume_skills"]
                        else "No listed skills detected."
                    )

                with col2:
                    st.subheader("Skills Detected in Job")
                    st.write(
                        ", ".join(results["job_skills"])
                        if results["job_skills"]
                        else "No listed skills detected."
                    )

                st.subheader("Matched Skills")
                if results["matched_skills"]:
                    st.success(
                        ", ".join(results["matched_skills"])
                    )
                else:
                    st.write("No overlapping taxonomy skills detected.")

                st.subheader("Potential Skill Gaps")
                if results["missing_skills"]:
                    st.warning(
                        "Not detected in resume: "
                        + ", ".join(results["missing_skills"])
                    )
                elif results["job_skills"]:
                    st.success(
                        "All detected job skills were also detected "
                        "in the resume."
                    )
                else:
                    st.info(
                        "No taxonomy skills were detected in the job "
                        "description. Add specific skills for gap analysis."
                    )

                st.caption(
                    "Scores are experimental similarity measures, not "
                    "probabilities of hiring success. Semantic similarity "
                    "does not prove skill proficiency. Review candidates "
                    "fairly and manually."
                )

            except Exception as error:
                st.error(
                    "Analysis failed. Check the deployment logs "
                    "if this continues."
                )
                st.code(str(error))


# ---------------- DATASET ANALYSIS ----------------

with tab2:
    st.header("Synthetic Dataset Analysis")

    dataset_file = st.file_uploader(
        "Upload hiresense_500_synthetic_resumes.csv",
        type=["csv"],
        key="dataset_upload"
    )

    if dataset_file is not None:
        try:
            df = pd.read_csv(dataset_file)

            st.success(f"Loaded {len(df)} records.")

            metric1, metric2 = st.columns(2)
            metric1.metric("Total Records", len(df))
            metric2.metric("Columns", len(df.columns))

            st.subheader("Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)

            if "target_role" in df.columns:
                st.subheader("Candidates by Target Role")
                st.bar_chart(df["target_role"].value_counts())

            if "experience_years" in df.columns:
                st.subheader("Experience Distribution")
                st.bar_chart(
                    df["experience_years"].value_counts().sort_index()
                )

            st.subheader("Missing Values")
            missing_df = (
                df.isnull().sum()
                .rename("Missing Values")
                .to_frame()
            )
            st.dataframe(missing_df, use_container_width=True)

        except Exception as error:
            st.error(f"Could not read CSV: {error}")

    else:
        st.info(
            "Extract the dataset ZIP on your computer, then upload "
            "hiresense_500_synthetic_resumes.csv here."
        )


# ---------------- ABOUT ----------------

with tab3:
    st.header("About HireSense AI")

    st.markdown("""
    **Current features**

    - NLP-based skill extraction
    - TF-IDF baseline matching
    - Sentence Transformer semantic matching
    - Experimental combined matching score
    - Skill-gap analysis
    - Synthetic dataset exploration

    **Planned enhancements**

    - RAG-based recruitment question-answering chatbot
    - LLM-generated explanations grounded in retrieved evidence
    - PDF resume parsing
    - Candidate comparison and model evaluation dashboard
    """)

    st.warning(
        "Educational prototype using experimental scoring. "
        "Do not use it to make automated hiring decisions."
    )
 # ---------------- RAG RECRUITMENT CHATBOT ----------------

with st.expander("🤖 HireSense AI Recruitment Chatbot", expanded=True):
    st.write(
        "Ask questions about your resume dataset. "
        "This version retrieves relevant records without an API key."
    )

    uploaded_resumes = st.file_uploader(
        "Upload your resumes CSV",
        type=["csv"],
        key="rag_resumes"
    )

    uploaded_jobs = st.file_uploader(
        "Upload your job descriptions CSV (optional)",
        type=["csv"],
        key="rag_jobs"
    )

    question = st.text_input(
        "Ask a recruitment question",
        placeholder="Example: Find Python candidates with SQL skills"
    )

    if st.button("Search recruitment data"):
        if uploaded_resumes is None:
            st.warning("Please upload your resumes CSV first.")
        elif not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                resumes = pd.read_csv(uploaded_resumes)

                resumes["search_text"] = resumes.astype(str).agg(
                    " | ".join, axis=1
                )

                documents = resumes["search_text"].tolist()
                document_types = ["Resume"] * len(documents)

                if uploaded_jobs is not None:
                    jobs = pd.read_csv(uploaded_jobs)
                    jobs["search_text"] = jobs.astype(str).agg(
                        " | ".join, axis=1
                    )
                    documents.extend(jobs["search_text"].tolist())
                    document_types.extend(
                        ["Job description"] * len(jobs)
                    )

                if not documents:
                    st.warning("No records found in the uploaded files.")
                else:
                    with st.spinner("Finding relevant records..."):
                        model = load_semantic_model()

                        question_vector = model.encode(
                            [question],
                            normalize_embeddings=True
                        )

                        document_vectors = model.encode(
                            documents,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )

                        similarities = (
                            document_vectors @ question_vector[0]
                        )

                        top_indices = similarities.argsort()[::-1][:5]

                                        # -------- BASIC EVIDENCE-BASED ANSWERING --------

                    retrieved_candidates = []

                    for index in top_indices:
                        if document_types[index] != "Resume":
                            continue

                        row = resumes.iloc[index]

                        candidate_id = row.get(
                            "candidate_id",
                            row.get("candidateid", "Unknown")
                        )

                        role = row.get("target_role", "Not specified")

                        experience = row.get(
                            "experience_years",
                            row.get("experience", "Not specified")
                        )

                        skills = str(row.get(
                            "technical_skills",
                            row.get("skills", "")
                        ))

                        retrieved_candidates.append({
                            "Candidate ID": candidate_id,
                            "Target Role": role,
                            "Experience (years)": experience,
                            "Skills": skills,
                            "Similarity (%)": round(
                                float(similarities[index]) * 100, 1
                            )
                        })

                    st.subheader("💬 Recruitment Answer")

                                       question_lower = question.lower()

                    # Search every resume for explicitly listed skills.
                    if (
                        "both" in question_lower
                        and "python" in question_lower
                        and "sql" in question_lower
                    ):
                        st.subheader("🔎 Full-Dataset Skill Search")

                        all_resume_matches = resumes[
                            resumes["search_text"].str.contains(
                                r"\bpython\b", case=False, regex=True, na=False
                            )
                            & resumes["search_text"].str.contains(
                                r"\bsql\b", case=False, regex=True, na=False
                            )
                        ].copy()

                        st.write(
                            f"Found **{len(all_resume_matches)} resumes** "
                            "that explicitly mention both Python and SQL."
                        )

                        if not all_resume_matches.empty:
                            st.dataframe(
                                all_resume_matches.drop(
                                    columns=["search_text"],
                                    errors="ignore"
                                ),
                                hide_index=True,
                                use_container_width=True
                            )
                        else:
                            st.info(
                                "No resumes explicitly mention both skills."
                            )

                    if not retrieved_candidates:
                        st.info(
                            "No resume records were retrieved for this question."
                        )

                    elif (
                        "both" in question_lower
                        and "python" in question_lower
                        and "sql" in question_lower
                    ):
                        matches = [
                            candidate for candidate in retrieved_candidates
                            if "python" in candidate["Skills"].lower()
                            and "sql" in candidate["Skills"].lower()
                        ]

                        if matches:
                            st.write(
                                "Among the retrieved candidates, "
                                "these records list both Python and SQL:"
                            )
                            st.dataframe(
                                pd.DataFrame(matches),
                                hide_index=True,
                                use_container_width=True
                            )
                        else:
                            st.write(
                                "No retrieved resume explicitly lists both "
                                "Python and SQL in the displayed skills field."
                            )

                    elif (
                        "compare" in question_lower
                        or "comparison" in question_lower
                    ):
                        st.write(
                            "Comparison of the retrieved candidates:"
                        )
                        st.dataframe(
                            pd.DataFrame(retrieved_candidates[:3]),
                            hide_index=True,
                            use_container_width=True
                        )

                    else:
                        st.write(
                            f"Found {len(retrieved_candidates)} resume "
                            "record(s) among the retrieved results. "
                            "Here is a summary of the available evidence:"
                        )

                        st.dataframe(
                            pd.DataFrame(retrieved_candidates[:3]),
                            hide_index=True,
                            use_container_width=True
                        )

                    st.caption(
                        "Answers are generated using simple rules over "
                        "retrieved records, not an LLM. Only the retrieved "
                        "results are considered; missing information is "
                        "not inferred."
                    )
                    st.subheader("Relevant evidence")
                    st.caption(
                        "Top matching records, ranked by semantic similarity."
                    )

                    for rank, index in enumerate(top_indices, start=1):
                        record_text = documents[index]

                        st.markdown(
                            f"### {rank}. {document_types[index]}"
                        )
                        st.metric(
                            "Semantic similarity",
                            f"{similarities[index] * 100:.1f}%"
                        )

                        if document_types[index] == "Resume":
                            row = resumes.iloc[index]

                            col1, col2 = st.columns(2)

                            with col1:
                                st.write(
                                    "**Candidate ID:**",
                                    row.get(
                                        "candidate_id",
                                        row.get("candidateid", "Not available")
                                    )
                                )
                                st.write(
                                    "**Target role:**",
                                    row.get("target_role", "Not available")
                                )

                            with col2:
                                st.write(
                                    "**Experience:**",
                                    row.get(
                                        "experience_years",
                                        row.get("experience", "Not available")
                                    )
                                )
                                st.write(
                                    "**Skills:**",
                                    row.get(
                                        "technical_skills",
                                        row.get("skills", "See full record")
                                    )
                                )

                        with st.expander("View full record"):
                            st.write(record_text)

                        st.divider()

                    st.info(
                        "Similarity measures text relevance, not candidate "
                        "quality or hiring probability. Review records manually."
                    )

            except Exception as error:
                st.error(f"Could not search the uploaded data: {error}")

                # Combine each resume row into searchable text
                resumes["search_text"] = resumes.astype(str).agg(
                    " | ".join, axis=1
                )

                documents = resumes["search_text"].tolist()
                document_types = ["Resume"] * len(documents)

                if uploaded_jobs is not None:
                    jobs = pd.read_csv(uploaded_jobs)
                    jobs["search_text"] = jobs.astype(str).agg(
                        " | ".join, axis=1
                    )
                    documents.extend(jobs["search_text"].tolist())
                    document_types.extend(["Job description"] * len(jobs))

                if not documents:
                    st.warning("No records found in the uploaded files.")
                else:
                    with st.spinner("Finding relevant records..."):
                        model = load_semantic_model()

                        question_vector = model.encode(
                            [question],
                            normalize_embeddings=True
                        )

                        document_vectors = model.encode(
                            documents,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )

                        similarities = (
                            document_vectors @ question_vector[0]
                        )

                        top_indices = similarities.argsort()[::-1][:5]
                    st.subheader("Relevant evidence")
                    st.caption(
                        "Top matching records, ranked by semantic similarity."
                    )

                    for rank, index in enumerate(top_indices, start=1):
                        record_text = documents[index]

                        st.markdown(f"### {rank}. {document_types[index]}")
                        st.metric(
                            "Semantic similarity",
                            f"{similarities[index] * 100:.1f}%"
                        )

                        if document_types[index] == "Resume":
                            row = resumes.iloc[index] if index < len(resumes) else None

                            if row is not None:
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write(
                                        "**Candidate ID:**",
                                        row.get("candidate_id",
                                        row.get("candidateid", "Not available"))
                                    )
                                    st.write(
                                        "**Target role:**",
                                        row.get("target_role", "Not available")
                                    )

                                with col2:
                                    st.write(
                                        "**Experience:**",
                                        row.get("experience_years",
                                        row.get("experience", "Not available"))
                                    )
                                    st.write(
                                        "**Skills:**",
                                        row.get("technical_skills",
                                        row.get("skills", "See record below"))
                                    )

                        with st.expander("View full record"):
                            st.write(record_text)

                        st.divider()

                    st.info(
                        "Similarity measures text relevance, not candidate "
                        "quality or hiring probability. Verify the original "
                        "records before drawing conclusions."
                    )

            except Exception as error:
                st.error(f"Could not search the uploaded data: {error}")
