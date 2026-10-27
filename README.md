# HireSense AI — AI Recruitment & Skill Intelligence Platform

An AI-powered recruitment intelligence project that analyzes synthetic resumes, matches candidates with job descriptions, extracts skills, and identifies skill gaps.

## Features

- Resume and job description analysis
- Automatic skill extraction using Python
- Resume matching using TF-IDF
- Semantic similarity using Sentence Transformers
- Candidate skill-gap analysis
- Data science and exploratory data analysis
- RAG-based document retrieval (planned development)

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF and cosine similarity
- Sentence Transformers
- Google Colab
- Git and GitHub

## Dataset

This project uses synthetic data for educational purposes:

- 500 synthetic candidate profiles
- 50 synthetic job descriptions
- 500 synthetic resume-job pairs

## Project Structure

```text
HireSense-AI/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── data/
│   ├── hiresense_500_synthetic_resumes.csv
│   ├── hiresense_50_job_descriptions.csv
│   └── hiresense_500_resume_job_pairs.csv
└── notebooks/
    └── HireSense_AI_Colab.ipynb
```

## Getting Started

1. Download or clone this repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Open the project notebook in Google Colab.
4. Upload or extract the synthetic CSV datasets.
5. Run the notebook cells in order.

## Disclaimer

This is an educational prototype. The dataset is synthetic, and matching scores are experimental. Do not use this project to make real hiring decisions.
