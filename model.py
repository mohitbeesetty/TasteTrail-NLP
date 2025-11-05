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

        try:
             self.nlp = spacy.load("en_core_web_sm") 
        except:
             print("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
             self.nlp = None
        
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("vader_lexicon", quiet=True)
        
        self.sia = SentimentIntensityAnalyzer()

    def _map_sentiment_to_score(self, sentiment):
        """
        Maps categorical sentiment to a 0.0-1.0 numerical score with an extreme positive bias,
        aiming to stabilize the overall rating around 3.5/5.0.
        Negative (0.60), Neutral (0.90), Positive (1.0).
        """
        if sentiment == "positive":
            return 1.0
        elif sentiment == "neutral":
            return 0.90
        elif sentiment == "negative":
            return 0.60
        return 0.5

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

    def process_review(self, review_text, original_rating, doc=None):
        """
        Runs the full analysis pipeline on a single review text and calculates
        the individual normalized score (0.0 - 5.0).
        """
        processed_text = self.preprocess_text(review_text)
        
        sentiment, confidence = self.analyze_sentiment(processed_text)
        
        if doc is None and self.nlp:
            doc = self.nlp(review_text)
        
        authenticity_score = self.calculate_authenticity_score(review_text, doc)
        
        # --- Normalization---
        sentiment_mapped = self._map_sentiment_to_score(sentiment)
        
        # Weighted Sentiment (WS) = Mapped Sentiment * Confidence (0.0 - 1.0)
        weighted_sentiment = sentiment_mapped * confidence
        
        # Normalize the 1-5 star rating to a 0.0-1.0 scale
        normalized_original_rating = original_rating / 5.0
        
        r_indiv = (
            (0.30 * weighted_sentiment) +  # Bias from BERT sentiment
            (0.05 * authenticity_score) +  # Heuristic authenticity check
            (0.65 * normalized_original_rating)
        ) * 5.0
        
        # Cap the score at 5.0 and round
        normalized_rating = round(min(r_indiv, 5.0), 2)

        return {
            "sentiment": sentiment,
            "sentiment_confidence": confidence,
            "authenticity_score": authenticity_score,
            "normalized_rating": normalized_rating
        }

    def analyze_reviews_in_bulk(self, df):
        """
        Optimized method to analyze all reviews using SpaCy's nlp.pipe() 
        for efficient batch processing.
        """
        if 'caption' not in df.columns or 'rating' not in df.columns:
            print("Error: 'caption' or 'rating' column not found. Cannot proceed with bulk analysis.")
            return df

        if self.nlp:
            print("Running SpaCy nlp.pipe() for bulk tokenization/parsing...")
            docs = list(self.nlp.pipe(df['caption']))
        else:
            docs = [None] * len(df)

        print("Applying BERT sentiment, heuristic scoring, and incorporating original ratings...")
        # Pass the original 'rating' from the DataFrame to the process_review method
        results = [
            self.process_review(text, rating, doc) 
            for text, rating, doc in zip(df['caption'], df['rating'], docs)
        ]
        
        analysis_df = pd.DataFrame(results)
        final_df = pd.concat([df.reset_index(drop=True), analysis_df.reset_index(drop=True)], axis=1)

        return final_df

    def calculate_overall_normalized_rating(self, analyzed_df):
        """
        Calculates the final overall restaurant rating by averaging the individual
        'normalized_rating' scores across all reviews.
        """
        if 'normalized_rating' not in analyzed_df.columns:
            print("Error: 'normalized_rating' column not found. Run analyze_reviews_in_bulk first.")
            return None
        
        # Use mean() to calculate the overall average rating (0.0 - 5.0)
        overall_rating = analyzed_df['normalized_rating'].mean()
        return round(overall_rating, 2)
    

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

        # Calculate and print the single, overall normalized rating
        overall_rating = analyzer.calculate_overall_normalized_rating(final_df)
        print(f"\n--- OVERALL RESTAURANT RATING (Normalized with Star Rating Bias - 65/30/5) ---")
        if overall_rating is not None:
             print(f"Final Normalized Rating: {overall_rating}/5.0")
        else:
             print("Could not calculate final rating due to missing data.")

        print("\n--- Full Analysis Results (First 5 Rows) ---")
        print(final_df[["caption", "rating", "sentiment", "sentiment_confidence", "authenticity_score", "normalized_rating"]].head())

    except FileNotFoundError:
        print(f"Error: The input file '{csv_path}' was not found. Please ensure it is in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")