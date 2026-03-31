"""
ML Service - Model Loading & Inference
========================================
Singleton service that loads ML models once and provides prediction methods.
"""

import os
import re
import numpy as np
import joblib
import logging
from django.conf import settings
from sklearn.metrics.pairwise import cosine_similarity

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

logger = logging.getLogger(__name__)

DEFAULT_STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was', 'were',
}

NLTK_RESOURCES = [
    ('tokenizers/punkt', 'punkt'),
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
    ('corpora/omw-1.4', 'omw-1.4'),
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
]

NLTK_OPTIONAL_DOWNLOADS = ['punkt_tab', 'averaged_perceptron_tagger_eng']

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

SUPPORTED_PREDICTION_MODELS = {'logistic_regression', 'svm', 'bert'}


class MLService:
    """Singleton ML service for sentiment prediction."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.model = None
        self.svm_model = None
        self.vectorizer = None
        self.bert_pipeline = None
        self.model_loaded = False
        self.svm_model_loaded = False
        self.bert_model_loaded = False
        self._models_checked = False
        self._optional_models_checked = False
        self.stop_words = set(DEFAULT_STOP_WORDS)
        self.lemmatizer = WordNetLemmatizer()
        self.nltk_ready = False
        self._init_nlp_resources()

    def _ensure_models_loaded(self):
        """Load models lazily to keep startup resilient for non-inference commands."""
        if not self._models_checked:
            self._models_checked = True
            self._load_models()

    def _ensure_nltk_resource(self, resource_path, resource_name):
        """Ensure an NLTK resource exists locally; attempt download if missing."""
        try:
            nltk.data.find(resource_path)
            return True
        except LookupError:
            try:
                return bool(nltk.download(resource_name, quiet=True))
            except Exception as exc:
                logger.warning("Could not download NLTK resource '%s': %s", resource_name, exc)
                return False

    def _init_nlp_resources(self):
        """Initialize tokenization/tagging resources with graceful fallbacks."""
        resources_ok = all(
            self._ensure_nltk_resource(path, name)
            for path, name in NLTK_RESOURCES
        )

        for optional in NLTK_OPTIONAL_DOWNLOADS:
            try:
                nltk.download(optional, quiet=True)
            except Exception:
                # Optional resources are non-fatal.
                pass

        if resources_ok:
            try:
                self.stop_words = set(stopwords.words('english'))
                self.nltk_ready = True
            except LookupError:
                logger.warning("NLTK stopwords unavailable; using minimal fallback stopword set.")
        else:
            logger.warning("Some NLTK resources are unavailable; using fallback tokenization where needed.")

    def _tokenize_words(self, text):
        try:
            return word_tokenize(text)
        except LookupError:
            return text.split()

    def _tokenize_sentences(self, text):
        try:
            return sent_tokenize(text)
        except LookupError:
            # Basic fallback sentence split when punkt is unavailable.
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    def _safe_pos_tag(self, tokens):
        try:
            return pos_tag(tokens)
        except LookupError:
            return [(token, '') for token in tokens]

    def _safe_lemmatize(self, token):
        if not self.nltk_ready:
            return token
        try:
            return self.lemmatizer.lemmatize(token)
        except LookupError:
            return token
    
    def _load_models(self):
        """Load ML models from disk."""
        model_dir = getattr(settings, 'ML_MODEL_DIR', None)
        if not model_dir:
            logger.warning("ML_MODEL_DIR not set in settings")
            return
        
        model_path = os.path.join(model_dir, 'logistic_regression.pkl')
        tfidf_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        
        if os.path.exists(model_path) and os.path.exists(tfidf_path):
            try:
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(tfidf_path)
                self.model_loaded = True
                logger.info("ML models loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load ML models: {e}")
        else:
            logger.warning(f"Model files not found in {model_dir}. "
                         "API will work in demo mode.")

        svm_path = os.path.join(model_dir, 'svm_model.pkl')
        if os.path.exists(svm_path):
            try:
                self.svm_model = joblib.load(svm_path)
                self.svm_model_loaded = True
                logger.info("SVM model loaded successfully")
            except Exception as exc:
                logger.warning("Failed to load SVM model: %s", exc)

    def _load_optional_models(self):
        """Load optional deep model only when explicitly needed."""
        if self._optional_models_checked:
            return

        self._optional_models_checked = True
        model_dir = getattr(settings, 'ML_MODEL_DIR', None)
        if not model_dir:
            return

        bert_path = os.path.join(model_dir, 'bert_sentiment')
        if not os.path.isdir(bert_path):
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

            tokenizer = AutoTokenizer.from_pretrained(bert_path)
            model = AutoModelForSequenceClassification.from_pretrained(bert_path)
            self.bert_pipeline = TextClassificationPipeline(
                model=model,
                tokenizer=tokenizer,
                top_k=None,
                truncation=True,
                max_length=512,
            )
            self.bert_model_loaded = True
            logger.info("Optional BERT model loaded for comparison")
        except Exception as exc:
            logger.warning("BERT model unavailable for runtime comparison: %s", exc)
    
    def _preprocess_text(self, text):
        """Clean and preprocess text."""
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = self._tokenize_words(text)
        # Remove stopwords and lemmatize
        tokens = [
            self._safe_lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return ' '.join(tokens)
    
    def predict(self, text):
        """
        Predict sentiment for a review.
        
        Returns:
            dict with sentiment, confidence, probabilities
        """
        self._ensure_models_loaded()

        if not self.model_loaded:
            # Demo mode - return a mock prediction based on simple heuristics
            return self._demo_predict(text)
        
        return self._predict_with_vector_model(text, self.model)

    def predict_with_model(self, text, model_name='logistic_regression'):
        """Predict sentiment using a specific model with safe fallbacks."""
        requested_model = (model_name or 'logistic_regression').lower().strip()
        if requested_model not in SUPPORTED_PREDICTION_MODELS:
            requested_model = 'logistic_regression'

        baseline = self.predict(text)
        baseline_response = {
            **baseline,
            'model_requested': requested_model,
            'model_used': 'logistic_regression',
            'fallback': requested_model != 'logistic_regression' or not self.model_loaded,
        }

        if requested_model == 'logistic_regression':
            return {
                **baseline,
                'model_requested': requested_model,
                'model_used': 'logistic_regression',
                'fallback': not self.model_loaded,
            }

        if requested_model == 'svm':
            self._ensure_models_loaded()
            if self.model_loaded and self.svm_model_loaded and self.svm_model and self.vectorizer:
                svm_result = self._predict_with_vector_model(text, self.svm_model)
                return {
                    **svm_result,
                    'model_requested': requested_model,
                    'model_used': 'svm',
                    'fallback': False,
                }

            return {
                **baseline_response,
                'reason': 'SVM model is not available in current runtime',
            }

        if requested_model == 'bert':
            self._load_optional_models()
            if self.bert_model_loaded and self.bert_pipeline:
                try:
                    bert_result = self._predict_with_bert(text)
                    return {
                        **bert_result,
                        'model_requested': requested_model,
                        'model_used': 'bert',
                        'fallback': False,
                    }
                except Exception as exc:
                    return {
                        **baseline_response,
                        'reason': f'BERT inference failed: {exc}',
                    }

            return {
                **baseline_response,
                'reason': 'BERT runtime dependencies/model files are unavailable',
            }

        return baseline_response

    def _predict_with_vector_model(self, text, model):
        """Run inference for models that use the shared TF-IDF vectorizer."""
        clean_text = self._preprocess_text(text)
        text_tfidf = self.vectorizer.transform([clean_text])

        prediction = model.predict(text_tfidf)[0]

        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_tfidf)[0]
            positive_prob = float(probabilities[1])
            negative_prob = float(probabilities[0])
        else:
            score = float(model.decision_function(text_tfidf)[0])
            positive_prob = float(1.0 / (1.0 + np.exp(-score)))
            negative_prob = 1.0 - positive_prob

        sentiment = 'positive' if prediction == 1 else 'negative'
        confidence = max(positive_prob, negative_prob)

        return {
            'sentiment': sentiment,
            'confidence': round(float(confidence), 4),
            'positive_prob': round(float(positive_prob), 4),
            'negative_prob': round(float(negative_prob), 4),
        }

    def _predict_with_bert(self, text):
        """Run optional BERT inference when transformers runtime is available."""
        if not self.bert_pipeline:
            raise RuntimeError('BERT pipeline is not loaded')

        outputs = self.bert_pipeline(text[:3000])
        labels = outputs[0] if outputs else []

        positive_prob = 0.5
        negative_prob = 0.5
        for item in labels:
            label = str(item.get('label', '')).lower()
            score = float(item.get('score', 0.0))
            if 'pos' in label or label.endswith('1'):
                positive_prob = score
            elif 'neg' in label or label.endswith('0'):
                negative_prob = score

        total = positive_prob + negative_prob
        if total > 0:
            positive_prob /= total
            negative_prob /= total

        sentiment = 'positive' if positive_prob >= negative_prob else 'negative'
        confidence = max(positive_prob, negative_prob)

        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'positive_prob': round(positive_prob, 4),
            'negative_prob': round(negative_prob, 4),
        }

    def compare_models(self, text):
        """Compare predictions across available LR, SVM, and BERT models."""
        base = self.predict(text)
        self._ensure_models_loaded()
        self._load_optional_models()

        models = {
            'logistic_regression': {
                **base,
                'available': True,
                'fallback': not self.model_loaded,
            }
        }

        if self.model_loaded and self.svm_model_loaded and self.svm_model and self.vectorizer:
            svm_prediction = self._predict_with_vector_model(text, self.svm_model)
            models['svm'] = {
                **svm_prediction,
                'available': True,
                'fallback': False,
            }
        else:
            models['svm'] = {
                **base,
                'available': False,
                'fallback': True,
                'reason': 'SVM model file not available in runtime',
            }

        if self.bert_model_loaded and self.bert_pipeline:
            try:
                bert_prediction = self._predict_with_bert(text)
                models['bert'] = {
                    **bert_prediction,
                    'available': True,
                    'fallback': False,
                }
            except Exception as exc:
                models['bert'] = {
                    **base,
                    'available': False,
                    'fallback': True,
                    'reason': f'BERT inference failed: {exc}',
                }
        else:
            models['bert'] = {
                **base,
                'available': False,
                'fallback': True,
                'reason': 'BERT runtime dependencies are not installed on server',
            }

        available_models = {
            name: values
            for name, values in models.items()
            if values.get('available', False)
        }
        winner = max(available_models, key=lambda model_name: available_models[model_name]['confidence'])

        return {
            'review': text,
            'winner': winner,
            'models': models,
        }

    def similarity_scores(self, source_text, candidate_texts):
        """Return cosine similarity scores between a source review and candidates."""
        if not candidate_texts:
            return []

        self._ensure_models_loaded()
        if self.model_loaded and self.vectorizer:
            corpus = [self._preprocess_text(source_text)] + [self._preprocess_text(item) for item in candidate_texts]
            tfidf_matrix = self.vectorizer.transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            return [float(score) for score in similarity]

        source_tokens = set(self._tokenize_words(source_text.lower()))
        scores = []
        for candidate in candidate_texts:
            candidate_tokens = set(self._tokenize_words(candidate.lower()))
            union_size = len(source_tokens | candidate_tokens)
            score = len(source_tokens & candidate_tokens) / union_size if union_size else 0.0
            scores.append(float(score))
        return scores
    
    def _demo_predict(self, text):
        """Demo prediction using simple keyword matching when model isn't loaded."""
        text_lower = text.lower()
        
        positive_words = {'amazing', 'great', 'excellent', 'wonderful', 'fantastic',
                         'brilliant', 'superb', 'outstanding', 'love', 'loved',
                         'beautiful', 'perfect', 'best', 'masterpiece', 'enjoy'}
        negative_words = {'terrible', 'awful', 'horrible', 'worst', 'bad', 'boring',
                         'waste', 'poor', 'disappointing', 'mediocre', 'hate',
                         'hated', 'stupid', 'pathetic', 'trash'}
        
        pos_count = sum(1 for w in text_lower.split() if w in positive_words)
        neg_count = sum(1 for w in text_lower.split() if w in negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return {
                'sentiment': 'positive',
                'confidence': 0.55,
                'positive_prob': 0.55,
                'negative_prob': 0.45,
            }
        
        pos_ratio = pos_count / total
        sentiment = 'positive' if pos_ratio >= 0.5 else 'negative'
        confidence = max(pos_ratio, 1 - pos_ratio)
        
        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'positive_prob': round(pos_ratio, 4),
            'negative_prob': round(1 - pos_ratio, 4),
        }
    
    def explain(self, text, num_features=10):
        """
        Get LIME explanation for a prediction.
        
        Returns word-level importance highlights.
        """
        self._ensure_models_loaded()

        if not self.model_loaded:
            return self._demo_explain(text)
        
        try:
            from lime.lime_text import LimeTextExplainer
            
            explainer = LimeTextExplainer(
                class_names=['Negative', 'Positive'],
                split_expression=r'\s+',
                random_state=42
            )
            
            def predict_fn(texts):
                tfidf_matrix = self.vectorizer.transform(
                    [self._preprocess_text(t) for t in texts]
                )
                return self.model.predict_proba(tfidf_matrix)
            
            explanation = explainer.explain_instance(
                text, predict_fn,
                num_features=num_features,
                num_samples=500
            )
            
            # Get overall prediction
            prediction = self.predict(text)
            predicted_class = 1 if prediction['sentiment'] == 'positive' else 0

            available_labels = list(explanation.local_exp.keys())
            if predicted_class not in available_labels and available_labels:
                logger.warning(
                    "Requested LIME label %s missing; using available label %s",
                    predicted_class,
                    available_labels[0],
                )
                predicted_class = available_labels[0]

            if predicted_class in explanation.local_exp:
                word_weights = explanation.as_list(label=predicted_class)
            else:
                logger.warning("LIME returned no local explanations; falling back to basic prediction response.")
                return {**prediction, 'explanation': [], 'text_highlights': []}
            
            highlights = []
            for word, weight in word_weights:
                direction = 'positive' if weight > 0 else 'negative'
                if predicted_class == 0:
                    direction = 'negative' if weight > 0 else 'positive'
                
                highlights.append({
                    'word': word,
                    'weight': round(abs(float(weight)), 4),
                    'direction': direction,
                })
            
            # Build per-word highlighting for the text
            word_map = {item['word'].lower(): item for item in highlights}
            text_highlights = []
            for word in text.split():
                clean = word.lower().strip('.,!?;:"\'()[]{}')
                if clean in word_map:
                    info = word_map[clean]
                    text_highlights.append({
                        'word': word,
                        'color': 'green' if info['direction'] == 'positive' else 'red',
                        'intensity': min(info['weight'] * 5, 1.0),
                        'direction': info['direction'],
                    })
                else:
                    text_highlights.append({
                        'word': word,
                        'color': 'neutral',
                        'intensity': 0,
                        'direction': 'neutral',
                    })
            
            return {
                **prediction,
                'explanation': highlights,
                'text_highlights': text_highlights,
            }
        except ImportError:
            logger.warning("LIME not installed, returning basic prediction")
            return {**self.predict(text), 'explanation': [], 'text_highlights': []}
    
    def _demo_explain(self, text):
        """Demo explanation when model isn't loaded."""
        prediction = self._demo_predict(text)
        
        positive_words = {'amazing', 'great', 'excellent', 'wonderful', 'fantastic',
                         'brilliant', 'superb', 'love', 'beautiful', 'perfect', 'best',
                         'masterpiece', 'enjoy', 'outstanding'}
        negative_words = {'terrible', 'awful', 'horrible', 'worst', 'bad', 'boring',
                         'waste', 'poor', 'disappointing', 'hate', 'stupid', 'trash'}
        
        highlights = []
        text_highlights = []
        
        for word in text.split():
            clean = word.lower().strip('.,!?;:"\'()[]{}')
            if clean in positive_words:
                item = {'word': word, 'weight': 0.8, 'direction': 'positive'}
                highlights.append(item)
                text_highlights.append({
                    'word': word, 'color': 'green',
                    'intensity': 0.8, 'direction': 'positive',
                })
            elif clean in negative_words:
                item = {'word': word, 'weight': 0.8, 'direction': 'negative'}
                highlights.append(item)
                text_highlights.append({
                    'word': word, 'color': 'red',
                    'intensity': 0.8, 'direction': 'negative',
                })
            else:
                text_highlights.append({
                    'word': word, 'color': 'neutral',
                    'intensity': 0, 'direction': 'neutral',
                })
        
        return {
            **prediction,
            'explanation': highlights,
            'text_highlights': text_highlights,
        }
    
    def analyze_aspects(self, text):
        """
        Aspect-based sentiment analysis.
        
        Returns per-aspect sentiment breakdown.
        """
        # Overall sentiment
        overall = self.predict(text)
        
        # Split into sentences
        sentences = self._tokenize_sentences(text)
        
        aspect_results = {}
        sentence_analyses = []
        
        for sentence in sentences:
            # Extract aspects via POS tagging
            tokens = self._tokenize_words(sentence)
            tagged = self._safe_pos_tag(tokens)
            
            aspects = []
            for word, tag in tagged:
                word_lower = word.lower()
                if tag.startswith('NN') and word_lower in MOVIE_ASPECTS:
                    aspects.append(word_lower)
            
            # Get sentiment for this sentence
            sent_result = self.predict(sentence)
            
            sentence_analyses.append({
                'sentence': sentence,
                'aspects': aspects,
                'sentiment': sent_result['sentiment'],
                'confidence': sent_result['confidence'],
            })
            
            # Map aspects to sentiments
            for aspect in aspects:
                if aspect not in aspect_results or sent_result['confidence'] > aspect_results[aspect]['confidence']:
                    aspect_results[aspect] = {
                        'aspect': aspect,
                        'sentiment': sent_result['sentiment'],
                        'confidence': sent_result['confidence'],
                    }
        
        aspects_list = sorted(
            aspect_results.values(),
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        return {
            'overall': overall,
            'aspects': aspects_list,
            'sentences': sentence_analyses,
        }


# Global instance
ml_service = MLService()
