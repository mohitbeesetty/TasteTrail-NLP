import pandas as pd
import numpy as np
import re
import os 

# Machine Learning & NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class TasteTrailReviewAnalyzer:
    def __init__(self):
        # BERT Model Initialization
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
            self.model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model=self.model, 
                tokenizer=self.tokenizer,
                max_length=512, 
                truncation=True
            )
            print("BERT Sentiment Model Loaded Successfully.")
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            self.sentiment_pipeline = None

        # Traditional NLP Tools (spaCy, NLTK)
        try:
             self.nlp = spacy.load("en_core_web_sm") 
        except:
             print("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
             self.nlp = None
        
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("vader_lexicon", quiet=True)
        
        self.sia = SentimentIntensityAnalyzer()

    def preprocess_text(self, text):
        """
        Cleans and normalizes the text by removing punctuation, lowercasing, and removing stopwords.
        """
        text = re.sub(r"[^\w\s]", "", str(text).lower().strip()) 
        stop_words = set(stopwords.words("english"))
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words]
        return " ".join(tokens)

    def calculate_authenticity_score(self, review_text, doc):
        """
        Calculates a heuristic 'authenticity' score (0.00 to 1.00) based on review features.
        Accepts the pre-processed 'doc' object from SpaCy's nlp.pipe().
        """
        if not self.nlp:
            return 0.0
        
        length_score = min(len(review_text.split()) / 100, 1.0) 
        
        food_terms = ["taste", "flavor", "texture", "dish", "menu", "portion", "service", "waiter", "host", "price", "ambiance"]
        food_term_score = sum(1 for term in food_terms if term in review_text.lower()) / len(food_terms)
        
        personal_markers = ["i", "we", "my", "our"]
        personal_count = sum(1 for token in doc if token.text.lower() in personal_markers)
        personal_score = min(personal_count / 10, 1.0) 
        
        sentiment_scores = self.sia.polarity_scores(review_text)
        emotion_score = abs(sentiment_scores["compound"]) 
        
        # Weighted average of heuristics
        weights = [0.3, 0.3, 0.2, 0.2] 
        final_score = sum([
            length_score * weights[0],
            food_term_score * weights[1],
            personal_score * weights[2],
            emotion_score * weights[3],
        ])
        return round(final_score, 2)

    def analyze_sentiment(self, review_text):
        """
        Performs sentiment analysis using BERT and maps 1-5 star output to
        'negative', 'neutral', or 'positive'.
        """
        if not self.sentiment_pipeline:
             return "error", 0.0

        bert_result = self.sentiment_pipeline(review_text)[0]
        label = bert_result["label"]
        score = bert_result["score"]

        # Map 1-5 star labels to categories
        if "1" in label or "2" in label:
            sentiment = "negative"
        elif "3" in label:
            sentiment = "neutral"
        else: 
            sentiment = "positive"

        return sentiment, round(score, 3)

    def process_review(self, review_text, doc=None):
        """
        Runs the full analysis pipeline on a single review text.
        Accepts an optional pre-parsed SpaCy 'doc' object.
        """
        processed_text = self.preprocess_text(review_text)
        
        sentiment, confidence = self.analyze_sentiment(processed_text)
        
        if doc is None and self.nlp:
            doc = self.nlp(review_text)
        
        authenticity_score = self.calculate_authenticity_score(review_text, doc)
        
        return {
            "sentiment": sentiment,
            "sentiment_confidence": confidence,
            "authenticity_score": authenticity_score,
        }

    def analyze_reviews_in_bulk(self, df):
        """
        Optimized method to analyze all reviews using SpaCy's nlp.pipe() 
        for efficient batch processing.
        """
        if 'caption' not in df.columns:
            print("Error: 'caption' column not found. Cannot proceed with bulk analysis.")
            return df

        if self.nlp:
            print("Running SpaCy nlp.pipe() for bulk tokenization/parsing...")
            docs = list(self.nlp.pipe(df['caption']))
        else:
            docs = [None] * len(df)

        print("Applying BERT sentiment and heuristic scoring...")
        results = [self.process_review(text, doc) for text, doc in zip(df['caption'], docs)]
        
        analysis_df = pd.DataFrame(results)
        final_df = pd.concat([df.reset_index(drop=True), analysis_df.reset_index(drop=True)], axis=1)

        return final_df

if __name__ == "__main__":
    print("Initializing TasteTrail Review Analyzer...")
    analyzer = TasteTrailReviewAnalyzer()
    
    csv_path = "reviews.csv"
    output_dir = "data"
    output_path = os.path.join(output_dir, "reviews_analyzed.csv")

    try:
        review_df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(review_df)} reviews from {csv_path}.")

        final_df = analyzer.analyze_reviews_in_bulk(review_df)
            
        os.makedirs(output_dir, exist_ok=True)
        final_df.to_csv(output_path, index=False)
        
        print(f"\nAnalysis complete. Processed {len(final_df)} reviews.")
        print(f"Enriched data saved to: {output_path}")

        print("\n--- Full Analysis Results (First 5 Rows) ---")
        print(final_df[["caption", "rating", "sentiment", "sentiment_confidence", "authenticity_score"]].head())

    except FileNotFoundError:
        print(f"Error: The input file '{csv_path}' was not found. Please ensure it is in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")