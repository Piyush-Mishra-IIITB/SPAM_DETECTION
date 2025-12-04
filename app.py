import streamlit as st
import nltk
import pickle

import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import PyPDF2

# ----------------------------
# Ensure required NLTK data is downloaded (deployment-friendly)
# ----------------------------
nltk_packages = ["punkt", "stopwords"]
for pkg in nltk_packages:
    try:
        if pkg == "punkt":
            nltk.data.find(f'tokenizers/{pkg}')
        else:
            nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

# ----------------------------
# Initialize
# ----------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="📧 Spam Classifier",
    page_icon="📧",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #4B0082;
        font-size: 40px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: gray;
        font-size: 18px;
        margin-bottom: 40px;
    }
    .card {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="title">📧 SMS/Email/PDF Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect whether a message or document is spam using ML</div>', unsafe_allow_html=True)

# ----------------------------
# Sidebar for PDF
# ----------------------------
st.sidebar.header("📄 PDF Spam Checker")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if st.sidebar.button("Check PDF"):
    if uploaded_file is None:
        st.sidebar.warning("Please upload a PDF file.")
    else:
        pdf_text = ""
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() + " "

        # Transform text
        transform_sms = " ".join([
            ps.stem(word) for word in nltk.word_tokenize(pdf_text.lower())
            if word.isalnum() and word not in stop_words and word not in string.punctuation
        ])
        vector = vectorizer.transform([transform_sms])
        result = model.predict(vector)[0]

        # Display result card
        if result == 1:
            st.markdown(f"""
                <div class="card" style="background-color:#FF6347; color:white;">
                <h2 style="text-align:center;">🚫 Spam Detected in PDF!</h2>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="card" style="background-color:#32CD32; color:white;">
                <h2 style="text-align:center;">✅ PDF Not Spam</h2>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("**Transformed PDF Text:**")
        st.write(transform_sms)

# ----------------------------
# Main area for text message
# ----------------------------
st.markdown("### ✉️ Text Message Spam Checker")
input_sms = st.text_area("Type your message here...", height=150)

def predict_text(text):
    transform_sms = " ".join([
        ps.stem(word) for word in nltk.word_tokenize(text.lower())
        if word.isalnum() and word not in stop_words and word not in string.punctuation
    ])
    vector = vectorizer.transform([transform_sms])
    result = model.predict(vector)[0]

    # Display result card
    if result == 1:
        st.markdown(f"""
            <div class="card" style="background-color:#FF6347; color:white;">
            <h2 style="text-align:center;">🚫 Spam Detected!</h2>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="card" style="background-color:#32CD32; color:white;">
            <h2 style="text-align:center;">✅ Not Spam</h2>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("**Transformed Text:**")
    st.write(transform_sms)

# Text prediction button
if st.button("Check Text Message"):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        predict_text(input_sms)
