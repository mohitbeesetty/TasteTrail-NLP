# 🍽️ Restaurant Review Analyzer

This module performs **sentiment analysis** and **authenticity scoring** for restaurant reviews.

### Features
- Sentiment prediction
- Authenticity scoring using linguistic and emotional heuristics

### Schema
# 🍽️ Restaurant Review Analyzer

This module performs automated analysis of restaurant reviews and produces:

- Overall sentiment (positive / neutral / negative) with a confidence score
- An authenticity score (0.00 - 1.00) indicating how likely the review is to be a genuine, detailed account
- Aspect-level scores (food quality, service, ambiance, value)

The code lives in `model.py` and exposes a `RestaurantReviewAnalyzer` class you can use programmatically or via the included script.

## Schema
The analyzer outputs documents that match the following schema (suitable for MongoDB):

```json
{
  "review_id": ObjectId,
  "restaurant_id": ObjectId,
  "user_id": ObjectId,
  "rating": Number,
  "review_text": String,
  "source": String,
  "authenticity_score": Number,
  "sentiment": String,
  "sentiment_confidence": Number,
  "timestamp": Date
}
```

## Quick start — setup and run

These steps will create a Python virtual environment, install required packages, download NLP resources and run the analyzer on the provided CSV (`TasteTrail-Scraper/data/reviews.csv`). All commands are for Windows PowerShell (adjust for your shell if needed).

1) Create & activate a virtual environment

```powershell
cd C:\Users\Mohit\Coding\TasteTrail\TasteTrail-NLP
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install packages

```powershell
pip install -r requirements.txt
```

3) Download NLP resources (spaCy model and NLTK data)

The repository includes `setup_resources.py` which fetches required NLTK corpora and the spaCy English model. Run:

```powershell
python setup_resources.py
```

4) Configure MongoDB connection (optional)

If you want the script to write results to MongoDB, copy the `.env.example` to `.env` and set `MONGODB_URI` and `DB_NAME`:

```
MONGODB_URI=mongodb://localhost:27017/
DB_NAME=tastetrail
```

If no MongoDB is available the analyzer still runs and returns results in Python.

5) Run the analyzer script

Run the model script which will process a CSV (default path in the script points to the scraper output):

```powershell
# run with the virtualenv python to ensure correct environment
C:/Users/Mohit/Coding/TasteTrail/.venv/Scripts/python.exe model.py
```

The script will load the pre-trained BERT sentiment model (this downloads ~600MB the first time), analyze reviews, and either update the configured MongoDB collection (`reviews`) or return results.

## Using the analyzer from Python

You can import and use the analyzer in other scripts:

```python
from model import RestaurantReviewAnalyzer

analyzer = RestaurantReviewAnalyzer()
result = analyzer.process_review("The pasta was amazing and the service was friendly.")
print(result)
# {
#   'sentiment': 'positive',
#   'sentiment_confidence': 0.93,
#   'authenticity_score': 0.68,
# }
```

To process a CSV programmatically:

```python
results = analyzer.process_reviews_from_csv('..\\TasteTrail-Scraper\\data\\reviews.csv')
```

## Notes and troubleshooting

- First-run downloads: the HuggingFace BERT model is large (~600–700MB). Expect a few minutes to download on first run.
- CPU only: by default the code runs on CPU. For faster inference, configure PyTorch to use a GPU if available.
- spaCy model: if you see "spaCy model 'en_core_web_sm' not found" run `python -m spacy download en_core_web_sm` or run `setup_resources.py`.
- NLTK resources: `setup_resources.py` downloads required NLTK corpora (punkt, stopwords, vader_lexicon, etc.). If you get LookupError for a specific NLTK resource, run `nltk.download('resource_name')`.
- MongoDB: if MongoDB is misconfigured the code will raise a connection error when attempting to write — either correct the URI in `.env` or modify `model.py` to skip DB writes.

## Output and where results go

- By default `model.py` updates documents in the `reviews` collection of the database set in `.env` (it matches on `review_id`).
- If you call `process_review` or `process_reviews_from_csv` from Python you receive results back as Python dictionaries / lists which you can save anywhere (CSV, database, etc.).

## Example run summary

- Downloads: HuggingFace model (~600–700MB) and spaCy model (~12MB).
- Runtime: depends on CPU/GPU and number of reviews. Processing a few hundred short reviews on a modern CPU may take several minutes.

## Files of interest

- `model.py` — main analyzer implementation
- `setup_resources.py` — helper to download NLP resources
- `requirements.txt` — Python dependencies
- `.env.example` — example MongoDB configuration