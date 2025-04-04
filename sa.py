import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import plotly.express as px
import plotly.graph_objects as go
import urllib.request
from urllib.error import URLError
from PyPDF2 import PdfReader
import io

# Try to import newspaper3k with error handling
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    st.warning("""
    The newspaper3k package is not installed. URL analysis features will be disabled.
    To enable URL analysis, please install the package using:
    ```
    pip install newspaper3k
    ```
    """)

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')
except Exception as e:
    st.error(f"Error downloading NLTK resources: {str(e)}")
    st.info("Please make sure you have an active internet connection and try again.")

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Comprehensive Sentiment Analysis Tool")
st.markdown("""
This app performs sentiment analysis on text using multiple methods and provides detailed insights.
""")

# Sidebar for input options
analysis_options = ["Single Text Analysis", "Bulk Text Analysis"]
if NEWSPAPER_AVAILABLE:
    analysis_options.append("URL Analysis")

st.sidebar.header("Analysis Options")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    analysis_options
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Function to extract text from TXT
def extract_text_from_txt(txt_file):
    try:
        return txt_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Error reading TXT file: {str(e)}")
        return None

# Function to extract article text from URL
def extract_article_text(url):
    if not NEWSPAPER_AVAILABLE:
        st.error("URL analysis is not available. Please install newspaper3k package.")
        return None
        
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        return {
            'title': article.title,
            'text': article.text,
            'summary': article.summary,
            'keywords': article.keywords,
            'publish_date': article.publish_date
        }
    except Exception as e:
        st.error(f"Error extracting article: {str(e)}")
        return None

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Sentiment analysis functions
def get_textblob_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def get_vader_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores

def get_sentiment_label(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Main content
if analysis_type == "Single Text Analysis":
    st.header("Single Text Analysis")
    text_input = st.text_area("Enter your text here:", height=150)
    
    if st.button("Analyze"):
        if text_input:
            # Preprocess text
            processed_text = preprocess_text(text_input)
            
            # Get sentiment scores
            textblob_polarity, textblob_subjectivity = get_textblob_sentiment(text_input)
            vader_scores = get_vader_sentiment(text_input)
            
            # Create columns for results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("TextBlob Analysis")
                st.write(f"Polarity: {textblob_polarity:.2f}")
                st.write(f"Subjectivity: {textblob_subjectivity:.2f}")
                st.write(f"Sentiment: {get_sentiment_label(textblob_polarity)}")
                
                # Polarity gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=textblob_polarity,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Polarity Score"},
                    gauge={'axis': {'range': [-1, 1]}}
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("VADER Analysis")
                st.write(f"Positive: {vader_scores['pos']:.2f}")
                st.write(f"Neutral: {vader_scores['neu']:.2f}")
                st.write(f"Negative: {vader_scores['neg']:.2f}")
                st.write(f"Compound: {vader_scores['compound']:.2f}")
                
                # Sentiment distribution pie chart
                fig = px.pie(
                    values=[vader_scores['pos'], vader_scores['neu'], vader_scores['neg']],
                    names=['Positive', 'Neutral', 'Negative'],
                    title='Sentiment Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                st.subheader("Text Analysis")
                st.write("Processed Text:")
                st.write(processed_text)
        
        else:
            st.warning("Please enter some text to analyze.")

elif analysis_type == "URL Analysis":
    st.header("URL Analysis")
    url_input = st.text_input("Enter the URL of the article:")
    
    if st.button("Analyze URL"):
        if url_input:
            with st.spinner("Extracting article content..."):
                article_data = extract_article_text(url_input)
                
                if article_data:
                    st.subheader("Article Information")
                    st.write(f"**Title:** {article_data['title']}")
                    if article_data['publish_date']:
                        st.write(f"**Published Date:** {article_data['publish_date']}")
                    
                    st.subheader("Article Summary")
                    st.write(article_data['summary'])
                    
                    st.subheader("Key Topics")
                    st.write(", ".join(article_data['keywords']))
                    
                    # Get sentiment scores
                    textblob_polarity, textblob_subjectivity = get_textblob_sentiment(article_data['text'])
                    vader_scores = get_vader_sentiment(article_data['text'])
                    
                    # Create columns for results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("TextBlob Analysis")
                        st.write(f"Polarity: {textblob_polarity:.2f}")
                        st.write(f"Subjectivity: {textblob_subjectivity:.2f}")
                        st.write(f"Sentiment: {get_sentiment_label(textblob_polarity)}")
                        
                        # Polarity gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=textblob_polarity,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Polarity Score"},
                            gauge={'axis': {'range': [-1, 1]}}
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("VADER Analysis")
                        st.write(f"Positive: {vader_scores['pos']:.2f}")
                        st.write(f"Neutral: {vader_scores['neu']:.2f}")
                        st.write(f"Negative: {vader_scores['neg']:.2f}")
                        st.write(f"Compound: {vader_scores['compound']:.2f}")
                        
                        # Sentiment distribution pie chart
                        fig = px.pie(
                            values=[vader_scores['pos'], vader_scores['neu'], vader_scores['neg']],
                            names=['Positive', 'Neutral', 'Negative'],
                            title='Sentiment Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not extract article content. Please check the URL and try again.")
        else:
            st.warning("Please enter a URL to analyze.")

else:  # Bulk Text Analysis
    st.header("Bulk Text Analysis")
    file_type = st.radio("Select file type:", ["TXT", "PDF"])
    
    if file_type == "TXT":
        uploaded_file = st.file_uploader("Upload a TXT file", type=['txt'])
    else:
        uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    
    if uploaded_file is not None:
        # Extract text based on file type
        if file_type == "TXT":
            text = extract_text_from_txt(uploaded_file)
        else:
            text = extract_text_from_pdf(uploaded_file)
        
        if text:
            # Split text into paragraphs or sections
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            
            # Create a DataFrame to store results
            results = []
            for i, para in enumerate(paragraphs):
                if para.strip():
                    # Preprocess text
                    processed_text = preprocess_text(para)
                    
                    # Get sentiment scores
                    textblob_polarity, textblob_subjectivity = get_textblob_sentiment(para)
                    vader_scores = get_vader_sentiment(para)
                    
                    results.append({
                        'Paragraph': i + 1,
                        'Text': para[:100] + '...' if len(para) > 100 else para,
                        'TextBlob_Polarity': textblob_polarity,
                        'TextBlob_Subjectivity': textblob_subjectivity,
                        'VADER_Positive': vader_scores['pos'],
                        'VADER_Neutral': vader_scores['neu'],
                        'VADER_Negative': vader_scores['neg'],
                        'VADER_Compound': vader_scores['compound'],
                        'Sentiment': get_sentiment_label(textblob_polarity)
                    })
            
            df = pd.DataFrame(results)
            
            # Display results
            st.subheader("Analysis Results")
            st.dataframe(df)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Distribution")
                fig = px.histogram(df, x='Sentiment', title='Sentiment Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Polarity vs Subjectivity")
                fig = px.scatter(
                    df,
                    x='TextBlob_Polarity',
                    y='TextBlob_Subjectivity',
                    color='Sentiment',
                    title='Polarity vs Subjectivity',
                    hover_data=['Text']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results",
                data=csv,
                file_name='sentiment_analysis_results.csv',
                mime='text/csv'
            )
        else:
            st.error("Could not extract text from the file. Please try again.")
    else:
        st.info(f"Please upload a {file_type} file to begin analysis.")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit") 
