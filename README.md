# AdaptLearn AI

### Explainable Adaptive Learning & Personalized Knowledge Intelligence

AdaptLearn AI is an AI-powered adaptive learning system that transforms course materials into a personalized learning experience. It combines semantic retrieval, Retrieval-Augmented Generation (RAG), adaptive assessment, machine-learning-based learner analytics, personalized recommendations, and reassessment.

## Problem

Traditional Learning Management Systems often provide static learning content without continuously adapting to individual learner understanding, knowledge gaps, or learning progress.

## Solution

AdaptLearn creates a continuous adaptive learning cycle:

**Predict → Diagnose → Retrieve → Personalize → Learn → Reassess → Improve**

Students can upload their own course material, interact with an AI tutor, take dynamically generated quizzes, identify weak concepts, receive personalized learning support, and reassess their understanding.

## Key Features

- Learner profile
- Course PDF upload and processing
- AI-powered knowledge navigator
- Semantic retrieval from uploaded content
- Adaptive quiz generation
- Knowledge-gap detection
- Weak-concept identification
- XGBoost-based learner analytics
- Personalized learning recommendations
- Targeted reassessment
- Learning improvement analysis

## System Architecture

Course PDF  
↓  
PDF Text Extraction  
↓  
SBERT Embeddings  
↓  
FAISS Semantic Retrieval  
↓  
RAG + Gemini  
↓  
Adaptive Assessment  
↓  
Knowledge Gap Detection  
↓  
XGBoost Learning Analytics  
↓  
Personalized Learning Support  
↓  
Targeted Reassessment  
↓  
Improvement Analysis

## Machine Learning

The learning-analytics component uses **XGBoost** with features derived from the **Open University Learning Analytics Dataset (OULAD)**.

The trained multiclass model predicts:

- Distinction
- Pass
- Fail
- Withdrawn

The prototype achieved approximately **70% classification accuracy**.

The XGBoost prediction provides broader learner analytics, while the adaptive quiz identifies immediate concept-level weaknesses.

## RAG Pipeline

AdaptLearn uses:

**PDF → Text Chunks → Sentence-BERT → FAISS → Relevant Context → Gemini**

Sentence-BERT converts learning content into semantic embeddings. FAISS performs similarity-based retrieval, and the retrieved course context is supplied to Gemini to generate grounded responses.

## Technology Stack

- Python
- Streamlit
- XGBoost
- Sentence-BERT
- FAISS
- Gemini
- PyPDF
- Pandas
- NumPy
- GitHub
- Streamlit Community Cloud

## Adaptive Learning Workflow

1. Create learner profile
2. Upload course material
3. Ask questions using the AI tutor
4. Generate an adaptive quiz
5. Identify knowledge gaps and weak concepts
6. Analyze learner outcome using XGBoost
7. Generate personalized learning support
8. Conduct targeted reassessment
9. Measure learning improvement

## Future Scope

- Persistent learner accounts and databases
- Long-term learner activity tracking
- Multi-course support
- Instructor analytics dashboard
- More extensive adaptive assessment
- Integration with institutional LMS platforms

## Project Status

Working prototype developed and deployed using Streamlit.