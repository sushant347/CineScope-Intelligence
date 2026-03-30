"""
Explainable AI (XAI) Module
============================
LIME explanations for sentiment predictions
Highlights which words contribute to positive/negative sentiment
"""

import os
import numpy as np
import joblib
from lime.lime_text import LimeTextExplainer


class SentimentExplainer:
    """LIME-based explainer for sentiment predictions."""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.explainer = LimeTextExplainer(
            class_names=['Negative', 'Positive'],
            split_expression=r'\s+',
            random_state=42
        )
        self._load_model()
    
    def _load_model(self):
        """Load the baseline model and vectorizer."""
        model_path = os.path.join(self.model_dir, 'logistic_regression.pkl')
        tfidf_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
        
        if os.path.exists(model_path) and os.path.exists(tfidf_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(tfidf_path)
        else:
            raise FileNotFoundError(
                f"Model files not found in {self.model_dir}. "
                "Train the baseline model first!"
            )
    
    def _predict_proba(self, texts):
        """Predict probabilities for LIME."""
        text_tfidf = self.vectorizer.transform(texts)
        return self.model.predict_proba(text_tfidf)
    
    def explain(self, text, num_features=10, num_samples=500):
        """
        Generate LIME explanation for a review.
        
        Args:
            text: Review text
            num_features: Number of top features to return
            num_samples: Number of perturbation samples
        
        Returns:
            dict with explanation data
        """
        # Get prediction
        proba = self._predict_proba([text])[0]
        predicted_class = int(np.argmax(proba))
        sentiment = 'positive' if predicted_class == 1 else 'negative'
        confidence = float(max(proba))
        
        # Generate LIME explanation
        explanation = self.explainer.explain_instance(
            text,
            self._predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=(predicted_class,)
        )
        
        # Extract word contributions
        word_weights = explanation.as_list(label=predicted_class)
        
        words = []
        for word, weight in word_weights:
            direction = 'positive' if weight > 0 else 'negative'
            # If predicted negative, flip the interpretation
            if predicted_class == 0:
                direction = 'negative' if weight > 0 else 'positive'
            
            words.append({
                'word': word,
                'weight': round(abs(float(weight)), 4),
                'direction': direction,
                'contribution': 'supports' if weight > 0 else 'opposes',
                'raw_weight': round(float(weight), 4),
            })
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_prob': float(proba[1]),
            'negative_prob': float(proba[0]),
            'explanation': words,
            'html': explanation.as_html(),
        }
    
    def get_word_highlights(self, text, num_features=15):
        """
        Get word-level highlights for UI display.
        
        Returns list of {word, color, intensity} for visualization.
        """
        result = self.explain(text, num_features=num_features)
        
        # Create word → highlight mapping
        word_map = {}
        for item in result['explanation']:
            word_map[item['word'].lower()] = {
                'direction': item['direction'],
                'intensity': min(item['weight'] * 5, 1.0),  # Normalize to 0-1
            }
        
        # Build highlighted text
        highlights = []
        for word in text.split():
            clean_word = word.lower().strip('.,!?;:"\'()[]{}')
            if clean_word in word_map:
                highlights.append({
                    'word': word,
                    'color': 'green' if word_map[clean_word]['direction'] == 'positive' else 'red',
                    'intensity': word_map[clean_word]['intensity'],
                    'direction': word_map[clean_word]['direction'],
                })
            else:
                highlights.append({
                    'word': word,
                    'color': 'neutral',
                    'intensity': 0,
                    'direction': 'neutral',
                })
        
        return {
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'highlights': highlights,
        }


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'models')
    
    explainer = SentimentExplainer(model_dir)
    
    test_reviews = [
        "This movie was absolutely amazing! The acting was superb and the storyline kept me engaged throughout.",
        "Terrible film. The plot was predictable and the acting was wooden. Complete waste of time.",
        "The cinematography was beautiful but the story was boring and too long.",
    ]
    
    for review in test_reviews:
        print(f"\nReview: \"{review[:80]}...\"")
        result = explainer.explain(review)
        print(f"Sentiment: {result['sentiment']} ({result['confidence']:.2%})")
        print("Top contributing words:")
        for word_info in result['explanation'][:5]:
            emoji = "🟢" if word_info['direction'] == 'positive' else "🔴"
            print(f"  {emoji} {word_info['word']}: {word_info['weight']:.4f} ({word_info['direction']})")
