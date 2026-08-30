import joblib
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from xgboost import XGBClassifier
from google import genai
from datetime import date
import textwrap
# --------------------------------------------------
# Load trained XGBoost model and label encoder
# --------------------------------------------------
xgb_model = XGBClassifier()
xgb_model.load_model("xgboost_model (1).json")
label_encoder = joblib.load("label_encoder.pkl")
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"])
# --------------------------------------------------
# Streamlit page setup
# --------------------------------------------------
st.set_page_config(
    page_title="AdaptLearn AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ---------- ACTIVE DAY TRACKING ----------
today = str(date.today())
if "active_dates" not in st.session_state:
    st.session_state["active_dates"] = []
if today not in st.session_state["active_dates"]:
    st.session_state["active_dates"].append(today)
# --------------------------------------------------
# Shared helper functions
# --------------------------------------------------
@st.cache_resource
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")
def split_text(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
# ---------- PREMIUM ADAPTLEARN THEME ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #F7F3EE;
    color: #172033;
}
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 4rem;
    max-width: 1180px;
}
[data-testid="stHeader"] {
    background: rgba(247,243,238,.92) !important;
}

[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 0;
}

[data-testid="stSidebar"] * {
    color: #F9FAFB !important;
}

[data-testid="stSidebar"] button {
    background: rgba(255,255,255,.07) !important;
    border: 1px solid rgba(255,255,255,.09) !important;
    border-radius: 14px !important;
    color: #F9FAFB !important;
    transition: all .2s ease;
}

[data-testid="stSidebar"] button:hover {
    background: #E95D4E !important;
    border-color: #E95D4E !important;
    transform: translateX(3px);
}

h1, h2, h3 {
    color: #172033;
    letter-spacing: -0.025em;
}

p, label {
    color: #374151;
}

/* Streamlit action buttons */
div.stButton > button,
[data-testid="stFormSubmitButton"] button {
    background: #E95D4E;
    color: #FFFFFF !important;
    border: 0;
    border-radius: 999px;
    padding: .72rem 1.15rem;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(233,93,78,.18);
    transition: transform .2s ease, box-shadow .2s ease, background .2s ease;
}

div.stButton > button *,
[data-testid="stFormSubmitButton"] button * {
    color: #FFFFFF !important;
}

div.stButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: #D94B3D;
    color: #FFFFFF !important;
    transform: translateY(-2px);
    box-shadow: 0 12px 26px rgba(233,93,78,.24);
}

/* Hero */
.premium-hero {
    position: relative;
    overflow: hidden;
    min-height: 410px;
    border-radius: 34px;
    padding: 58px 56px;
    background:
        radial-gradient(circle at 85% 20%, rgba(255,205,153,.72), transparent 26%),
        radial-gradient(circle at 72% 72%, rgba(233,93,78,.22), transparent 30%),
        linear-gradient(135deg, #FFF8F1 0%, #F8E8DE 100%);
    border: 1px solid rgba(23,32,51,.07);
    box-shadow: 0 22px 55px rgba(23,32,51,.08);
    margin: 8px 0 30px 0;
}

.hero-kicker {
    display: inline-block;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(233,93,78,.10);
    color: #C9473A !important;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: 20px;
}

.premium-hero h1 {
    max-width: 690px;
    font-size: clamp(46px, 6vw, 76px);
    line-height: .98;
    margin: 0;
    color: #172033;
    font-weight: 800;
    letter-spacing: -.055em;
}

.premium-hero h1 span {
    color: #E95D4E;
}

.premium-hero p {
    max-width: 610px;
    margin-top: 24px;
    font-size: 18px;
    line-height: 1.65;
    color: #5D6472;
}

.hero-orb {
    position: absolute;
    width: 210px;
    height: 210px;
    right: 55px;
    top: 76px;
    border-radius: 50%;
    background: #172033;
    box-shadow: inset -30px -24px 0 rgba(255,255,255,.05), 0 24px 45px rgba(23,32,51,.20);
}

.hero-orb:before {
    content: "AI";
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    color: #FFF8F1;
    font-size: 62px;
    font-weight: 800;
}

.hero-orb:after {
    content: "";
    position: absolute;
    width: 54px;
    height: 54px;
    border-radius: 50%;
    background: #E95D4E;
    right: -12px;
    bottom: 16px;
}

/* Section heading */
.section-eyebrow {
    color: #E95D4E !important;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .12em;
    font-size: 12px;
    margin-bottom: 8px;
}

.section-title {
    font-size: 36px;
    font-weight: 800;
    color: #172033;
    letter-spacing: -.04em;
    margin-bottom: 18px;
}

/* Sliding card rail */
.slider-shell {
    overflow: hidden;
    width: 100%;
    padding: 10px 0 26px 0;
    mask-image: linear-gradient(to right, transparent, black 4%, black 96%, transparent);
    -webkit-mask-image: linear-gradient(to right, transparent, black 4%, black 96%, transparent);
}

.slider-track {
    display: flex;
    gap: 18px;
    width: max-content;
    animation: adaptScroll 30s linear infinite;
}

.slider-shell:hover .slider-track {
    animation-play-state: paused;
}

.slide-card {
    width: 285px;
    min-height: 230px;
    padding: 26px;
    border-radius: 26px;
    background: #FFFFFF;
    border: 1px solid rgba(23,32,51,.08);
    box-shadow: 0 14px 34px rgba(23,32,51,.07);
    transition: transform .25s ease, box-shadow .25s ease;
}

.slide-card:hover {
    transform: translateY(-7px);
    box-shadow: 0 20px 42px rgba(23,32,51,.12);
}

.slide-card.dark {
    background: #172033;
}

.slide-card.coral {
    background: #E95D4E;
}

.slide-card.sand {
    background: #F1D8C7;
}

.slide-card .number {
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .12em;
    color: #E95D4E;
}

.slide-card h3 {
    margin: 52px 0 10px;
    font-size: 24px;
    color: #172033;
}

.slide-card p {
    color: #697080;
    line-height: 1.55;
}

.slide-card.dark h3, .slide-card.dark p, .slide-card.dark .number,
.slide-card.coral h3, .slide-card.coral p, .slide-card.coral .number {
    color: #FFFFFF !important;
}

.slide-card.sand .number {
    color: #9B443B;
}

@keyframes adaptScroll {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}

/* Dark intelligence section */
.intelligence {
    margin: 28px 0;
    padding: 42px;
    border-radius: 30px;
    background: #172033;
    color: white;
    box-shadow: 0 22px 48px rgba(23,32,51,.16);
}

.intelligence .mini {
    color: #F59B8F !important;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.intelligence h2 {
    color: #FFFFFF;
    font-size: 34px;
    margin: 8px 0 12px;
}

.intelligence p {
    color: #C9CED8;
    max-width: 760px;
    line-height: 1.65;
}

/* Journey pills */
.journey {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 18px 0 10px;
}

.journey span {
    background: #FFFFFF;
    color: #172033 !important;
    border: 1px solid rgba(23,32,51,.08);
    border-radius: 999px;
    padding: 10px 15px;
    font-weight: 700;
    box-shadow: 0 6px 16px rgba(23,32,51,.05);
}

/* Inputs / forms */
input, textarea {
    background: #FFFFFF !important;
    color: #172033 !important;
}

div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #172033 !important;
}

[data-testid="stForm"],
[data-testid="stExpander"],
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,.72) !important;
    border: 1px solid rgba(23,32,51,.08) !important;
    border-radius: 20px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important;
    color: #172033 !important;
    border: 1px dashed #D7B9AA !important;
}

[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(23,32,51,.08);
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 8px 22px rgba(23,32,51,.05);
}

[data-testid="stMetric"] * {
    color: #172033 !important;
}

[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
div[role="radiogroup"] label,
div[role="radiogroup"] label p,
div[role="radiogroup"] span,
[data-testid="stForm"] p,
[data-testid="stForm"] label,
[data-testid="stForm"] span,
[data-testid="stSlider"] p,
[data-testid="stSlider"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stExpander"] p,
[data-testid="stExpander"] span {
    color: #172033 !important;
}

[data-testid="stAlert"] {
    border-radius: 16px;
}

hr {
    border-color: rgba(23,32,51,.10);
}

/* Responsive */
@media (max-width: 850px) {
    .premium-hero {
        padding: 38px 28px;
        min-height: 390px;
    }
    .premium-hero h1 {
        font-size: 48px;
        max-width: 100%;
    }
    .premium-hero p {
        max-width: 100%;
        padding-right: 90px;
    }
    .hero-orb {
        width: 90px;
        height: 90px;
        right: 24px;
        top: 250px;
    }
    .hero-orb:before {
        font-size: 28px;
    }
}
</style>
""", unsafe_allow_html=True)
# ==================================================
# PAGE NAVIGATION
# ==================================================
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "🏠 Home"
def go_to_page(page_name):
    st.session_state["active_page"] = page_name
st.sidebar.title("🧠 AdaptLearn AI")
st.sidebar.caption(
    "Personalized Learning Intelligence"
)
st.sidebar.divider()
st.sidebar.button(
    "🏠 Home",
    use_container_width=True,
    on_click=go_to_page,
    args=("🏠 Home",)
)
st.sidebar.button(
    "📚 Learn",
    use_container_width=True,
    on_click=go_to_page,
    args=("📚 Learn",)
)
st.sidebar.button(
    "🧩 Adaptive Quiz",
    use_container_width=True,
    on_click=go_to_page,
    args=("🧩 Adaptive Quiz",)
)
st.sidebar.button(
    "🎯 Recommendations",
    use_container_width=True,
    on_click=go_to_page,
    args=("🎯 Recommendations",)
)
st.sidebar.divider()
st.sidebar.caption(
    "Upload → Learn → Assess → Improve"
)
page = st.session_state[
    "active_page"
]
st.sidebar.divider()
st.sidebar.info(
    "Learning Loop:\n\n"
    "Predict → Diagnose → Retrieve → Practice → Reassess → Adapt"
)
# ==================================================
# HOME PAGE
# ==================================================
if page == "🏠 Home":
    st.markdown(textwrap.dedent("""
    <div class="premium-hero">
        <div class="hero-kicker">Adaptive learning, built around you</div>
        <h1>Learn smarter.<br><span>Adapt faster.</span></h1>
        <p>
            Turn your own course material into an intelligent learning journey —
            ask, assess, diagnose, improve and reassess with AI.
        </p>
        <div class="hero-orb"></div>
    </div>
    """), unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.button(
            "Start Learning →",
            use_container_width=True,
            key="hero_learn",
            on_click=go_to_page,
            args=("📚 Learn",)
        )
    with c2:
        st.button(
            "Take a Quiz →",
            use_container_width=True,
            key="hero_quiz",
            on_click=go_to_page,
            args=("🧩 Adaptive Quiz",)
        )
    components.html(
    """
    <style>
        body {
            margin: 0;
            background: #F7F3EE;
            font-family: Arial, sans-serif;
            overflow: hidden;
        }
        .title-small {
            color: #E95D4E;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 2px;
        }
        .title {
            color: #172033;
            font-size: 34px;
            font-weight: 800;
            margin: 8px 0 22px;
        }
        .slider {
            width: 100%;
            overflow: hidden;
        }
        .track {
            display: flex;
            gap: 18px;
            width: max-content;
            animation: scroll 28s linear infinite;
        }
        .track:hover {
            animation-play-state: paused;
        }
        .card {
            width: 260px;
            min-width: 260px;
            height: 200px;
            padding: 25px;
            border-radius: 24px;
            background: white;
            box-sizing: border-box;
            box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        }
        .dark {
            background: #172033;
            color: white;
        }
        .coral {
            background: #E95D4E;
            color: white;
        }

        .sand {
            background: #F1D8C7;
        }

        .number {
            font-size: 12px;
            font-weight: bold;
            color: #E95D4E;
            letter-spacing: 1px;
        }

        .dark .number,
        .coral .number {
            color: white;
        }

        h3 {
            margin-top: 38px;
            margin-bottom: 8px;
            font-size: 22px;
        }

        p {
            font-size: 14px;
            line-height: 1.5;
            opacity: 0.8;
        }

        @keyframes scroll {
            from {
                transform: translateX(0);
            }

            to {
                transform: translateX(-50%);
            }
        }
    </style>

    <div class="title-small">YOUR LEARNING LOOP</div>
    <div class="title">One journey. Continuously adapting.</div>

    <div class="slider">
        <div class="track">

            <div class="card">
                <div class="number">01 · LEARN</div>
                <h3>Upload your PDF</h3>
                <p>Bring your own course material into AdaptLearn.</p>
            </div>

            <div class="card sand">
                <div class="number">02 · RETRIEVE</div>
                <h3>Ask the AI Tutor</h3>
                <p>SBERT and FAISS retrieve relevant course material.</p>
            </div>

            <div class="card dark">
                <div class="number">03 · ASSESS</div>
                <h3>Adaptive Quiz</h3>
                <p>Discover what you understand and where you need support.</p>
            </div>

            <div class="card coral">
                <div class="number">04 · PREDICT</div>
                <h3>ML Intelligence</h3>
                <p>XGBoost predicts the learner's broader learning outcome.</p>
            </div>

            <div class="card">
                <div class="number">05 · IMPROVE</div>
                <h3>Personalized Support</h3>
                <p>Receive targeted explanations for weak concepts.</p>
            </div>

            <div class="card sand">
                <div class="number">06 · REASSESS</div>
                <h3>Measure Progress</h3>
                <p>Reassess your understanding after personalized learning.</p>
            </div>

            <!-- DUPLICATE FOR CONTINUOUS SLIDE -->

            <div class="card">
                <div class="number">01 · LEARN</div>
                <h3>Upload your PDF</h3>
                <p>Bring your own course material into AdaptLearn.</p>
            </div>

            <div class="card sand">
                <div class="number">02 · RETRIEVE</div>
                <h3>Ask the AI Tutor</h3>
                <p>SBERT and FAISS retrieve relevant course material.</p>
            </div>

            <div class="card dark">
                <div class="number">03 · ASSESS</div>
                <h3>Adaptive Quiz</h3>
                <p>Discover what you understand and where you need support.</p>
            </div>

            <div class="card coral">
                <div class="number">04 · PREDICT</div>
                <h3>ML Intelligence</h3>
                <p>XGBoost predicts the learner's broader learning outcome.</p>
            </div>

            <div class="card">
                <div class="number">05 · IMPROVE</div>
                <h3>Personalized Support</h3>
                <p>Receive targeted explanations for weak concepts.</p>
            </div>

            <div class="card sand">
                <div class="number">06 · REASSESS</div>
                <h3>Measure Progress</h3>
                <p>Reassess your understanding after personalized learning.</p>
            </div>
        </div>
    </div>
    """,
    height=330,
    scrolling=False
)
st.markdown(textwrap.dedent("""
    <div class="section-eyebrow">How it works</div>
    <div class="journey">
        <span>📄 Upload</span>
        <span>→</span>
        <span>🤖 Learn</span>
        <span>→</span>
        <span>🧩 Assess</span>
        <span>→</span>
        <span>📈 Predict</span>
        <span>→</span>
        <span>🎯 Improve</span>
        <span>→</span>
        <span>🔄 Reassess</span>
    </div>
    """), unsafe_allow_html=True)
if "learner_score" in st.session_state:
        st.info(
            "You already have an assessment result. "
            "Open Recommendations to continue your personalized learning path."
        )
# ==================================================
# LEARNER PROFILE
# ==================================================
if page == "📚 Learn":
    st.header("👤 Learner Profile")
    st.caption(
        "Enter your details so AdaptLearn can personalize "
        "your learning experience."
    )
    if "student_name" not in st.session_state:
        st.session_state["student_name"] = ""
    if "student_email" not in st.session_state:
        st.session_state["student_email"] = ""
    with st.form("learner_profile_form"):
        student_name = st.text_input(
            "Name",
            value=st.session_state["student_name"],
            placeholder="Enter your name"
        )
        student_email = st.text_input(
            "Email (optional)",
            value=st.session_state["student_email"],
            placeholder="example@gmail.com"
        )
        save_profile = st.form_submit_button(
            "💾 Save Profile"
        )
    if save_profile:
        if student_name.strip() == "":
            st.warning("Please enter your name.")

        elif (
            student_email.strip() != ""
            and "@" not in student_email
        ):
            st.warning(
                "Please enter a valid email "
                "or leave it blank."
            )

        else:

            st.session_state["student_name"] = (
                student_name.strip()
            )

            st.session_state["student_email"] = (
                student_email.strip()
            )

            st.success(
                f"Welcome, "
                f"{st.session_state['student_name']}! 👋"
            )
    st.divider()
    # ==================================================
    # RAG KNOWLEDGE NAVIGATOR
    # ==================================================
    st.header("🤖 AI Knowledge Navigator")

    st.caption(
        "Upload course material and ask questions. "
        "The AI answers using the uploaded material."
    )

    uploaded_pdf = st.file_uploader(
        "📄 Upload Course PDF",
        type=["pdf"]
    )

    question = st.text_input(
        "💬 Ask a question from your course material",
        placeholder="Example: What is the difference between TCP and UDP?"
    )

    if st.button("✨ Ask AI Tutor"):

        if uploaded_pdf is None:
            st.warning("Please upload your course PDF first.")

        elif question.strip() == "":
            st.warning("Please enter a question.")

        else:

            # Load SBERT only when needed
            with st.spinner("Loading AI retrieval model..."):
                sbert_model = load_sbert()

            # Read PDF
            reader = PdfReader(uploaded_pdf)

            pdf_text = ""

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    pdf_text += text + "\n"

            # Check PDF text
            if pdf_text.strip() == "":

                st.error(
                    "No readable text was found in this PDF."
                )

            else:

                # Save PDF for Adaptive Assessment
                st.session_state["course_pdf_text"] = pdf_text
                st.session_state["course_pdf_name"] = uploaded_pdf.name

                # Split PDF into chunks
                pdf_chunks = split_text(pdf_text)

                # Create SBERT embeddings
                chunk_embeddings = sbert_model.encode(
                    pdf_chunks,
                    normalize_embeddings=True
                )

                chunk_embeddings = np.array(
                    chunk_embeddings
                ).astype("float32")

                # Create FAISS index
                pdf_index = faiss.IndexFlatIP(
                    chunk_embeddings.shape[1]
                )

                pdf_index.add(chunk_embeddings)

                # Convert learner question into embedding
                query_embedding = sbert_model.encode(
                    [question],
                    normalize_embeddings=True
                )

                query_embedding = np.array(
                    query_embedding
                ).astype("float32")

                # Retrieve top 3 relevant chunks
                scores, indices = pdf_index.search(
                    query_embedding,
                    k=min(3, len(pdf_chunks))
                )

                # Collect retrieved content
                retrieved_chunks = []

                for i in indices[0]:

                    if i >= 0:
                        retrieved_chunks.append(
                            pdf_chunks[i]
                        )

                context = "\n\n".join(
                    retrieved_chunks
                )

                # RAG Prompt
                rag_prompt = f"""
    You are AdaptLearn AI, an educational tutor.

    Answer the learner's question using ONLY the
    course material provided below.

    Do not use outside information.

    If the answer cannot be found in the provided
    material, say:

    "I could not find enough information in the
    uploaded course material to answer this question."

    Explain the answer in simple student-friendly language.

    COURSE MATERIAL:

    {context}

    LEARNER QUESTION:

    {question}

    Provide a clear and concise explanation.
    """

                # Generate answer using Gemini
                try:

                    with st.spinner(
                        "Generating personalized answer..."
                    ):

                        interaction = client.interactions.create(
                            model="gemini-3.6-flash",
                            input=rag_prompt
                        )

                    st.success(
                        "Answer generated from your course material."
                    )

                    st.subheader(
                        "🎓 AdaptLearn AI Tutor"
                    )

                    st.write(
                        interaction.output_text
                    )

                    # Show retrieved evidence
                    with st.expander(
                        "📚 View supporting course content"
                    ):

                        for rank, chunk in enumerate(
                            retrieved_chunks,
                            start=1
                        ):

                            st.markdown(
                                f"**Source {rank}**"
                            )

                            st.write(
                                chunk
                            )

                            st.caption(
                                "Semantic similarity: "
                                f"{scores[0][rank - 1]:.3f}"
                            )

                            st.divider()

                except Exception as e:

                    st.error(
                        "The AI answer could not be generated."
                    )

                    st.write(
                        "Technical details:",
                        str(e)
                    )

    st.divider()
# ==================================================
# QUIZ PAGE
# ==================================================
if page == "🧩 Adaptive Quiz":
        # ---------- LEARNER ACTIVITY TRACKING ----------

    if "quiz_attempts" not in st.session_state:
        st.session_state["quiz_attempts"] = 0

    if "total_interactions" not in st.session_state:
        st.session_state["total_interactions"] = 0

    if "quiz_scores" not in st.session_state:
        st.session_state["quiz_scores"] = []
# ==================================================
# AI ADAPTIVE ASSESSMENT
# ==================================================
    st.header("🧩 AI Adaptive Assessment")

    st.caption(
        "AdaptLearn automatically creates an assessment "
        "from your uploaded learning material."
    )
    # --------------------------------------------------
    # Check whether PDF has been processed
    # --------------------------------------------------
    if "course_pdf_text" not in st.session_state:

        st.info(
            "📄 Upload a PDF and ask at least one question "
            "in the Knowledge Navigator first."
        )

    else:

        pdf_name = st.session_state.get(
            "course_pdf_name",
            "Uploaded Course Material"
        )

        st.success(
            f"📚 Learning material loaded: {pdf_name}"
        )

        number_of_questions = st.select_slider(
            "Number of diagnostic questions",
            options=[5, 6, 7, 8, 9, 10],
            value=5
        )
        # --------------------------------------------------
        # Generate quiz
        # --------------------------------------------------
        if st.button("✨ Generate Assessment"):

            course_text = st.session_state[
                "course_pdf_text"
            ]

            # Limit text sent to Gemini
            course_text_for_quiz = course_text[:30000]

            quiz_prompt = f"""
    You are the assessment engine of AdaptLearn AI.

    Create exactly {number_of_questions} multiple-choice
    diagnostic questions using ONLY the course material
    provided below.

    The purpose is to measure how well the learner
    understands the uploaded material.
    Rules:
    1. Use ONLY information present in the course material.
    2. Cover different important concepts.
    3. Do not make all questions from one paragraph.
    4. Each question must have exactly four options.
    5. Only one option must be correct.
    6. Include conceptual questions rather than only
    memorization questions.
    7. Do NOT include "I don't know" as one of the
    four options. The application adds it separately.
    Return ONLY valid JSON.
    Use exactly this structure:
    [
    {{
        "question": "Question text",
        "options": [
            "Option A",
            "Option B",
            "Option C",
            "Option D"
        ],
        "answer": "Exact correct option",
        "concept": "Concept being tested"
    }}
    ]
    COURSE MATERIAL:
    {course_text_for_quiz}
    """
            try:

                with st.spinner(
                    "🧠 Creating assessment from your course material..."
                ):

                    interaction = client.interactions.create(
                        model="gemini-3.6-flash",
                        input=quiz_prompt
                    )

                    quiz_output = interaction.output_text

                    # Clean possible markdown formatting
                    quiz_output = quiz_output.strip()

                    if quiz_output.startswith("```json"):
                        quiz_output = quiz_output[7:]

                    elif quiz_output.startswith("```"):
                        quiz_output = quiz_output[3:]

                    if quiz_output.endswith("```"):
                        quiz_output = quiz_output[:-3]

                    import json

                    generated_quiz = json.loads(
                        quiz_output.strip()
                    )

                    st.session_state[
                        "generated_quiz"
                    ] = generated_quiz

                    st.success(
                        "Assessment generated successfully!"
                    )

            except Exception as e:

                st.error(
                    "Assessment could not be generated."
                )

                st.write(
                    "Technical details:",
                    str(e)
                )
    # ==================================================
    # DISPLAY GENERATED QUIZ
    # ==================================================
    if "generated_quiz" in st.session_state:

        quiz = st.session_state[
            "generated_quiz"
        ]

        st.subheader("📝 Diagnostic Quiz")

        responses = []

        with st.form("adaptive_quiz_form"):

            for i, q in enumerate(quiz):

                st.markdown(
                    f"### Question {i + 1}"
                )

                st.write(
                    q["question"]
                )

                options = (
                    q["options"]
                    + ["I don't know / Skip"]
                )

                selected = st.radio(
                    "Choose your answer",
                    options,
                    index=None,
                    key=f"adaptive_question_{i}"
                )

                responses.append(
                    selected
                )

                st.divider()

            submit_quiz = st.form_submit_button(
                "🧠 Analyze My Understanding"
            )
        # --------------------------------------------------
        # Analyze learner responses
        # --------------------------------------------------
        if submit_quiz:
                 # Record learner activity
            st.session_state["quiz_attempts"] += 1

            st.session_state["total_interactions"] += len(responses)
            correct = 0
            skipped = 0

            weak_concepts = []

            question_results = []

            total_questions = len(
                quiz
            )

            for i, response in enumerate(
                responses
            ):

                correct_answer = quiz[i][
                    "answer"
                ]

                concept = quiz[i].get(
                    "concept",
                    "Unknown concept"
                )

                if (
                    response is None
                    or response == "I don't know / Skip"
                ):

                    skipped += 1
                    
                    weak_concepts.append(
                        concept
                    )
                    result_status = "skipped"
                elif response == correct_answer:

                    correct += 1
                    result_status = "correct"
                else:

                    weak_concepts.append(
                        concept
                    )
                    result_status = "wrong"
                question_results.append(
    {
        "question": quiz[i]["question"],
        "selected_answer": response,
        "correct_answer": correct_answer,
        "concept": concept,
        "status": result_status
    }
)
                st.session_state[
    "question_results"
] = question_results
            score = (
                correct / total_questions
            ) * 100
            st.session_state["quiz_scores"].append(score)

            skip_rate = (
                skipped / total_questions
            ) * 100
            # ----------------------------------------------
            # Determine learning level
            # ----------------------------------------------
            if score < 50:

                gap = "High Gap"

                learning_level = "Beginner"

                recommendation = (
                    "Start with fundamentals and "
                    "simple explanations."
                )

            elif score < 70:

                gap = "Moderate Gap"

                learning_level = "Intermediate"

                recommendation = (
                    "Revise weak concepts and "
                    "study worked examples."
                )

            else:

                gap = "Low Gap"

                learning_level = "Advanced"

                recommendation = (
                    "Continue with advanced concepts "
                    "and challenging practice."
                )
            # Remove duplicate concepts
            weak_concepts = list(
                dict.fromkeys(
                    weak_concepts
                )
            )

            # Save learner profile
            st.session_state[
                "learner_score"
            ] = score

            st.session_state[
                "learner_gap"
            ] = gap

            st.session_state[
                "learning_level"
            ] = learning_level

            st.session_state[
                "weak_concepts"
            ] = weak_concepts
            # ----------------------------------------------
            # Display analysis
            # ----------------------------------------------
            st.success(
                "Assessment analyzed successfully."
            )

            st.subheader(
                "🧠 Your Learning Diagnosis"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Score",
                    f"{score:.0f}%"
                )

            with col2:

                st.metric(
                    "Correct",
                    f"{correct}/{total_questions}"
                )

            with col3:

                st.metric(
                    "Skipped",
                    skipped
                )

            with col4:

                st.metric(
                    "Knowledge Gap",
                    gap
                )

            st.write(
                f"**Current Learning Level:** "
                f"{learning_level}"
            )

            st.write(
                f"**Skip Rate:** "
                f"{skip_rate:.0f}%"
            )

            st.info(
                "🎯 Recommended Next Step: "
                + recommendation
            )
            # ----------------------------------------------
            # Weak concept detection
            # ----------------------------------------------
            if weak_concepts:

                st.subheader(
                    "⚠️ Concepts That Need Attention"
                )

                for concept in weak_concepts:

                    st.write(
                        f"• {concept}"
                    )

            else:

                st.success(
                    "🎉 No major weak concepts detected!"
                )
                st.divider()

st.subheader(
    "📋 Question-by-Question Result"
)

for i, result in enumerate(
    st.session_state[
        "question_results"
    ]
):

    st.markdown(
        f"### Question {i + 1}"
    )

    st.write(
        result["question"]
    )

    if result["status"] == "correct":

        st.success(
            f"✅ Your answer: "
            f"{result['selected_answer']}"
        )

    elif result["status"] == "wrong":

        st.error(
            f"❌ Your answer: "
            f"{result['selected_answer']}"
        )

        st.success(
            f"✅ Correct answer: "
            f"{result['correct_answer']}"
        )

    else:

        st.warning(
            "⚠️ You skipped this question."
        )

        st.success(
            f"✅ Correct answer: "
            f"{result['correct_answer']}"
        )

    st.caption(
        f"Concept: "
        f"{result['concept']}"
    )

    st.divider()
# ==================================================
# RECOMMENDATIONS PAGE
# ==================================================
if page == "🎯 Recommendations":

    st.header("🎯 Personalized Recommendations")
        # ==========================================
    # XGBOOST LEARNER ANALYTICS
    # ==========================================
    if st.session_state.get("quiz_scores"):

        # Real values collected from this LMS
        avg_score = sum(
            st.session_state["quiz_scores"]
        ) / len(st.session_state["quiz_scores"])

        assessments_attempted = st.session_state.get(
            "quiz_attempts",
            0
        )

        total_clicks = st.session_state.get(
            "total_interactions",
            0
        )

        active_days = len(
            st.session_state.get(
                "active_dates",
                []
            )
        )

        has_assessment = 1

        # Course/profile values required by
        # the OULAD-trained model
        num_of_prev_attempts = 0
        studied_credits = 60

        learner_features = pd.DataFrame(
            [[
                num_of_prev_attempts,
                studied_credits,
                avg_score,
                assessments_attempted,
                total_clicks,
                active_days,
                has_assessment
            ]],
            columns=[
                "num_of_prev_attempts",
                "studied_credits",
                "avg_score",
                "assessments_attempted",
                "total_clicks",
                "active_days",
                "has_assessment"
            ]
        )

        try:

            prediction = xgb_model.predict(
                learner_features
            )

            predicted_outcome = (
                label_encoder.inverse_transform(
                    prediction.astype(int)
                )[0]
            )

            st.session_state[
                "predicted_outcome"
            ] = predicted_outcome

            # ---------- DISPLAY ----------

            st.subheader(
                "🤖 ML Learning Analytics"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Predicted Outcome",
                    predicted_outcome
                )

            with col2:
                st.metric(
                    "Average Score",
                    f"{avg_score:.1f}%"
                )

            with col3:
                st.metric(
                    "Assessments",
                    assessments_attempted
                )

            # ---------- SUPPORT LEVEL ----------

            if predicted_outcome in [
                "Fail",
                "Withdrawn"
            ]:

                support_level = "High"

                st.warning(
                    "📚 Additional learning support "
                    "is recommended."
                )

            elif predicted_outcome == "Pass":

                support_level = "Moderate"

                st.info(
                    "🎯 Continue targeted practice "
                    "to strengthen understanding."
                )

            else:

                support_level = "Low"

                st.success(
                    "🌟 Strong learning pattern. "
                    "Continue with advanced practice."
                )

            st.session_state[
                "support_level"
            ] = support_level

            st.caption(
                "The XGBoost prediction uses "
                "assessment performance and "
                "learner activity patterns."
            )

        except Exception as e:

            st.error(
                f"ML prediction could not be generated: {e}"
            )

    else:

        st.info(
            "Complete an Adaptive Quiz to generate "
            "your ML learning analytics."
        )
    if "learner_score" not in st.session_state:

        st.info(
            "Complete the Adaptive Quiz first to "
            "generate personalized recommendations."
        )

    elif (
        "weak_concepts" in st.session_state
        and st.session_state["weak_concepts"]
    ):

        st.subheader("🎯 Personalized Learning")

        weak_concepts = st.session_state[
            "weak_concepts"
        ]

        learning_level = st.session_state.get(
            "learning_level",
            "Beginner"
        )

        st.write(
            "**Concepts selected for personalized learning:**"
        )

        for concept in weak_concepts:
            st.write(f"• {concept}")

        if st.button(
            "🚀 Start Personalized Learning"
        ):

            if (
                "course_pdf_text"
                not in st.session_state
            ):

                st.warning(
                    "Please upload and process "
                    "your course PDF first."
                )

            else:

                course_text = (
                    st.session_state[
                        "course_pdf_text"
                    ]
                )

                with st.spinner(
                    "Finding the best learning material "
                    "for your weak concepts..."
                ):

                    # Load SBERT
                    sbert_model = load_sbert()

                    # Split course material
                    learning_chunks = split_text(
                        course_text
                    )

                    # Create embeddings
                    learning_embeddings = (
                        sbert_model.encode(
                            learning_chunks,
                            normalize_embeddings=True
                        )
                    )

                    learning_embeddings = np.array(
                        learning_embeddings
                    ).astype("float32")

                    # Create FAISS index
                    learning_index = faiss.IndexFlatIP(
                        learning_embeddings.shape[1]
                    )

                    learning_index.add(
                        learning_embeddings
                    )

                    # Create search query using weak concepts
                    weak_query = (
                        "Explain these concepts: "
                        + ", ".join(weak_concepts)
                    )

                    query_embedding = sbert_model.encode(
                        [weak_query],
                        normalize_embeddings=True
                    )

                    query_embedding = np.array(
                        query_embedding
                    ).astype("float32")

                    # Retrieve relevant course sections
                    scores, indices = learning_index.search(
                        query_embedding,
                        k=min(5, len(learning_chunks))
                    )

                    personalized_chunks = []

                    for i in indices[0]:

                        if i >= 0:
                            personalized_chunks.append(
                                learning_chunks[i]
                            )

                    personalized_context = "\n\n".join(
                        personalized_chunks
                    )

                    # Adapt explanation style
                    if learning_level == "Beginner":

                        teaching_style = """
    Explain from the fundamentals.
    Use very simple language.
    Use an analogy if useful.
    Give one simple example.
    """

                    elif learning_level == "Intermediate":

                        teaching_style = """
    Give a concise revision.
    Explain the important relationships.
    Include a worked example where possible.
    """

                    else:

                        teaching_style = """
    Give a deeper explanation.
    Connect related concepts.
    Include a challenging application or example.
    """

                    personalized_prompt = f"""
    You are the personalized learning engine of
    AdaptLearn AI.

    The learner completed a diagnostic assessment.

    CURRENT LEARNING LEVEL:
    {learning_level}

    WEAK CONCEPTS:
    {", ".join(weak_concepts)}

    Use ONLY the uploaded course material given below.

    COURSE MATERIAL:
    {personalized_context}

    TEACHING INSTRUCTIONS:
    {teaching_style}

    For each weak concept:

    1. Explain the concept clearly.
    2. Explain why it is important.
    3. Give an example using the course material.
    4. Mention one key point the learner should remember.

    Do not introduce information that is not supported
    by the uploaded course material.

    Make the response easy for a student to study.
    """

                    try:

                        interaction = client.interactions.create(
                            model="gemini-3.6-flash",
                            input=personalized_prompt
                        )

                        st.session_state[
                            "personalized_learning"
                        ] = interaction.output_text

                    except Exception as e:

                        st.error(
                            "Personalized learning could not be generated."
                        )

                        st.write(
                            "Technical details:",
                            str(e)
                        )

    if "personalized_learning" in st.session_state:

        st.success(
            "Personalized learning path generated! 📚"
        )

        st.subheader(
            "🧠 Your Personalized Learning Material"
        )

        st.write(
            st.session_state[
                "personalized_learning"
            ]
        )

    # ==================================================
    # REASSESSMENT AFTER PERSONALIZED LEARNING
    # ==================================================

    if (
        "personalized_learning" in st.session_state
        and "weak_concepts" in st.session_state
        and st.session_state["weak_concepts"]
    ):

        st.divider()

        st.header("🔄 Personalized Reassessment")

        st.caption(
            "Test your understanding again after studying "
            "your personalized learning material."
        )

        if st.button("✨ Generate Reassessment"):

            weak_concepts = st.session_state[
                "weak_concepts"
            ]

            course_text = st.session_state.get(
                "course_pdf_text",
                ""
            )

            reassessment_prompt = f"""
    You are the reassessment engine of AdaptLearn AI.

    The learner previously had difficulty with:

    {", ".join(weak_concepts)}

    Create exactly 3 multiple-choice questions
    focused ONLY on these weak concepts.

    Use ONLY information from the uploaded
    course material below.

    COURSE MATERIAL:

    {course_text[:30000]}

    Rules:

    1. Create exactly 3 questions.
    2. Focus on the learner's weak concepts.
    3. Do not repeat the earlier questions exactly.
    4. Each question must have exactly four options.
    5. Only one option must be correct.
    6. Prefer conceptual understanding over memorization.
    7. Return ONLY valid JSON.

    Use exactly this format:

    [
    {{
        "question": "Question text",
        "options": [
            "Option A",
            "Option B",
            "Option C",
            "Option D"
        ],
        "answer": "Exact correct option",
        "concept": "Concept being tested"
    }}
    ]
    """

            try:

                with st.spinner(
                    "🧠 Creating your personalized reassessment..."
                ):

                    interaction = client.interactions.create(
                        model="gemini-3.6-flash",
                        input=reassessment_prompt
                    )

                    reassessment_output = (
                        interaction.output_text.strip()
                    )

                    if reassessment_output.startswith(
                        "```json"
                    ):
                        reassessment_output = (
                            reassessment_output[7:]
                        )

                    elif reassessment_output.startswith(
                        "```"
                    ):
                        reassessment_output = (
                            reassessment_output[3:]
                        )

                    if reassessment_output.endswith(
                        "```"
                    ):
                        reassessment_output = (
                            reassessment_output[:-3]
                        )

                    import json

                    reassessment_quiz = json.loads(
                        reassessment_output.strip()
                    )

                    st.session_state[
                        "reassessment_quiz"
                    ] = reassessment_quiz

                    # Remove old result when a new
                    # reassessment is generated
                    st.session_state.pop(
                        "reassessment_score",
                        None
                    )

                    st.session_state.pop(
                        "improvement",
                        None
                    )

                    st.success(
                        "Reassessment generated successfully!"
                    )

            except Exception as e:

                st.error(
                    "Reassessment could not be generated."
                )

                st.write(
                    "Technical details:",
                    str(e)
                )
                # ==================================================
    # DISPLAY AND ANALYZE REASSESSMENT
    # ==================================================

    if "reassessment_quiz" in st.session_state:

        reassessment_quiz = st.session_state[
            "reassessment_quiz"
        ]

        st.subheader(
            "📝 Check Your Improvement"
        )

        reassessment_responses = []

        with st.form(
            "reassessment_form"
        ):

            for i, q in enumerate(
                reassessment_quiz
            ):

                st.markdown(
                    f"### Reassessment Question {i + 1}"
                )

                st.write(
                    q["question"]
                )

                reassessment_options = (
                    q["options"]
                    + ["I don't know / Skip"]
                )

                selected = st.radio(
                    "Choose your answer",
                    reassessment_options,
                    index=None,
                    key=f"reassessment_question_{i}"
                )

                reassessment_responses.append(
                    selected
                )

                st.divider()

            submit_reassessment = (
                st.form_submit_button(
                    "📊 Analyze My Improvement"
                )
            )

        if submit_reassessment:

            reassessment_correct = 0
            reassessment_skipped = 0
            remaining_weak_concepts = []

            total_reassessment = len(
                reassessment_quiz
            )

            for i, response in enumerate(
                reassessment_responses
            ):

                correct_answer = (
                    reassessment_quiz[i][
                        "answer"
                    ]
                )

                concept = (
                    reassessment_quiz[i].get(
                        "concept",
                        "Unknown concept"
                    )
                )

                if (
                    response is None
                    or response
                    == "I don't know / Skip"
                ):

                    reassessment_skipped += 1

                    remaining_weak_concepts.append(
                        concept
                    )

                elif response == correct_answer:

                    reassessment_correct += 1

                else:

                    remaining_weak_concepts.append(
                        concept
                    )

            reassessment_score = (
                reassessment_correct
                / total_reassessment
            ) * 100

            initial_score = (
                st.session_state.get(
                    "learner_score",
                    0
                )
            )

            improvement = (
                reassessment_score
                - initial_score
            )

            remaining_weak_concepts = list(
                dict.fromkeys(
                    remaining_weak_concepts
                )
            )

            st.session_state[
                "reassessment_score"
            ] = reassessment_score

            st.session_state[
                "improvement"
            ] = improvement

            st.session_state[
                "remaining_weak_concepts"
            ] = remaining_weak_concepts
            # ==================================================
    # LEARNING PROGRESS REPORT
    # ==================================================

    if "reassessment_score" in st.session_state:

        st.divider()

        st.header(
            "📈 Learning Progress"
        )

        initial_score = st.session_state.get(
            "learner_score",
            0
        )

        reassessment_score = (
            st.session_state[
                "reassessment_score"
            ]
        )

        improvement = st.session_state[
            "improvement"
        ]

        remaining_weak = (
            st.session_state.get(
                "remaining_weak_concepts",
                []
            )
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Initial Score",
                f"{initial_score:.0f}%"
            )

        with col2:

            st.metric(
                "Reassessment Score",
                f"{reassessment_score:.0f}%"
            )

        with col3:

            st.metric(
                "Improvement",
                f"{improvement:+.0f}%"
            )

        if reassessment_score >= 70:

            st.success(
                "🎉 Strong improvement! "
                "You have achieved good understanding "
                "of the reassessed concepts."
            )

            next_step = (
                "Move to more advanced concepts "
                "or challenging practice."
            )

        elif reassessment_score >= 50:

            st.info(
                "👍 Your understanding is improving, "
                "but some concepts still need revision."
            )

            next_step = (
                "Review the remaining weak concepts "
                "and attempt another short practice."
            )

        else:

            st.warning(
                "📚 More support is recommended "
                "before moving to advanced material."
            )

            next_step = (
                "Revisit the personalized explanations "
                "with simpler examples and fundamentals."
            )

        st.subheader(
            "🎯 Recommended Next Learning Step"
        )

        st.info(
            next_step
        )

        if remaining_weak:

            st.subheader(
                "⚠️ Concepts Still Needing Attention"
            )

            for concept in remaining_weak:
                st.write(
                    f"• {concept}"
                )

        else:

            st.success(
                "✅ No major weak concepts remain "
                "in this reassessment."
            )

    st.divider()

