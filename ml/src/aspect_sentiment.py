"""
Aspect-Based Sentiment Analysis
=================================
Extract aspect nouns using POS tagging
Run sentiment per sentence to get per-aspect sentiment

Example:
  "Acting was great but story was boring"
  → Acting: Positive, Story: Negative
"""

import os
import re
import numpy as np
import joblib
import nltk
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

logger = logging.getLogger(__name__)

NLTK_RESOURCES = [
    ('tokenizers/punkt', 'punkt'),
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
]


def _ensure_nltk_resource(resource_path, resource_name):
    try:
        nltk.data.find(resource_path)
        return True
    except LookupError:
        try:
            return bool(nltk.download(resource_name, quiet=True))
        except Exception as exc:
            logger.warning("Could not download NLTK resource '%s': %s", resource_name, exc)
            return False


def _safe_sent_tokenize(text):
    try:
        return sent_tokenize(text)
    except LookupError:
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


def _safe_word_tokenize(text):
    try:
        return word_tokenize(text)
    except LookupError:
        return text.split()


def _safe_pos_tag(tokens):
    try:
        return pos_tag(tokens)
    except LookupError:
        return [(token, '') for token in tokens]

# Common movie-related aspects
MOVIE_ASPECTS = {
    'acting', 'actor', 'actress', 'performance', 'cast',
    'story', 'plot', 'storyline', 'narrative', 'script', 'writing',
    'direction', 'director', 'directing',
    'cinematography', 'visuals', 'effects', 'cgi', 'animation',
    'music', 'soundtrack', 'score', 'sound',
    'dialogue', 'humor', 'comedy', 'action', 'drama',
    'ending', 'beginning', 'pacing', 'editing',
    'characters', 'character', 'villain', 'hero', 'protagonist',
    'scenes', 'scene', 'setting', 'atmosphere',
    'movie', 'film', 'show', 'series',
}


class AspectSentimentAnalyzer:
    """Aspect-based sentiment analysis using POS tagging + sentence-level sentiment."""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.nltk_ready = all(
            _ensure_nltk_resource(path, name)
            for path, name in NLTK_RESOURCES
        )

        for optional in ('punkt_tab', 'averaged_perceptron_tagger_eng'):
            try:
                nltk.download(optional, quiet=True)
            except Exception:
                pass

        if not self.nltk_ready:
            logger.warning("NLTK resources are partially unavailable; using fallback tokenizers for aspect analysis.")

        self._load_model()
    
    def _load_model(self):
        """Load the baseline sentiment model."""
        model_path = os.path.join(self.model_dir, 'logistic_regression.pkl')
        tfidf_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
        
        if os.path.exists(model_path) and os.path.exists(tfidf_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(tfidf_path)
    
    def _predict_sentence_sentiment(self, sentence):
        """Predict sentiment for a single sentence."""
        if not self.model or not self.vectorizer:
            raise RuntimeError("Model not loaded")
        
        text_tfidf = self.vectorizer.transform([sentence.lower()])
        proba = self.model.predict_proba(text_tfidf)[0]
        predicted = int(np.argmax(proba))
        
        return {
            'sentiment': 'positive' if predicted == 1 else 'negative',
            'confidence': float(max(proba)),
            'positive_prob': float(proba[1]),
            'negative_prob': float(proba[0]),
        }
    
    def _extract_aspects(self, sentence):
        """Extract aspect nouns from a sentence using POS tagging."""
        tokens = _safe_word_tokenize(sentence)
        tagged = _safe_pos_tag(tokens)
        
        aspects = []
        for word, tag in tagged:
            word_lower = word.lower()
            # Look for nouns that are movie aspects
            if tag.startswith('NN') and word_lower in MOVIE_ASPECTS:
                aspects.append(word_lower)
            # Also check bigrams
            elif tag.startswith('NN') and len(word_lower) > 3:
                # Include general nouns that might be aspects
                aspects.append(word_lower)
        
        # Filter to only known movie aspects if any found
        known_aspects = [a for a in aspects if a in MOVIE_ASPECTS]
        if known_aspects:
            return known_aspects
        
        # If no known aspects, return the first noun found (if any)
        return aspects[:1] if aspects else []
    
    def analyze(self, text):
        """
        Perform aspect-based sentiment analysis.
        
        Args:
            text: Review text
        
        Returns:
            dict with overall sentiment and per-aspect breakdowns
        """
        # Overall sentiment
        overall = self._predict_sentence_sentiment(text)
        
        # Split into sentences
        sentences = _safe_sent_tokenize(text)
        
        # Analyze each sentence
        aspect_results = {}
        sentence_analyses = []
        
        for sentence in sentences:
            aspects = self._extract_aspects(sentence)
            sent_result = self._predict_sentence_sentiment(sentence)
            
            sentence_analyses.append({
                'sentence': sentence,
                'aspects': aspects,
                'sentiment': sent_result['sentiment'],
                'confidence': sent_result['confidence'],
            })
            
            # Map aspects to sentiments
            for aspect in aspects:
                if aspect not in aspect_results:
                    aspect_results[aspect] = {
                        'aspect': aspect,
                        'sentiment': sent_result['sentiment'],
                        'confidence': sent_result['confidence'],
                        'sentences': [sentence],
                    }
                else:
                    # Update with latest or higher confidence
                    if sent_result['confidence'] > aspect_results[aspect]['confidence']:
                        aspect_results[aspect]['sentiment'] = sent_result['sentiment']
                        aspect_results[aspect]['confidence'] = sent_result['confidence']
                    aspect_results[aspect]['sentences'].append(sentence)
        
        # Sort aspects by confidence
        aspects_list = sorted(
            aspect_results.values(),
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        return {
            'overall': overall,
            'aspects': aspects_list,
            'sentences': sentence_analyses,
            'num_aspects_found': len(aspects_list),
        }


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'models')
    
    analyzer = AspectSentimentAnalyzer(model_dir)
    
    test_reviews = [
        "The acting was great but the story was boring. The music was excellent though.",
        "Terrible direction and awful script. However, the cinematography was stunning.",
        "Amazing performance by the cast. The dialogue was witty and the pacing was perfect.",
    ]
    
    for review in test_reviews:
        print(f"\nReview: \"{review}\"")
        result = analyzer.analyze(review)
        print(f"Overall: {result['overall']['sentiment']} ({result['overall']['confidence']:.2%})")
        print("Aspects:")
        for aspect in result['aspects']:
            emoji = "✅" if aspect['sentiment'] == 'positive' else "❌"
            print(f"  {emoji} {aspect['aspect']}: {aspect['sentiment']} ({aspect['confidence']:.2%})")
