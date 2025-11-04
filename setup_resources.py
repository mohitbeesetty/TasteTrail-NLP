import nltk
import spacy

# Download all required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('all')  # This will download all NLTK resources

# Download spaCy model
spacy.cli.download("en_core_web_sm")