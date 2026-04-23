"""API Views - Sentiment Analysis Endpoints"""

import logging
import re
from collections import Counter
from datetime import timedelta

from django.db import DatabaseError
from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate
from django.http import Http404
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.utils.dateparse import parse_date
from rest_framework import generics, permissions, status
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
from rest_framework.views import APIView

from .ml_service import MOVIE_ASPECTS, ml_service
from .models import Prediction
from .serializers import (
    AspectInputSerializer,
    BatchPredictInputSerializer,
    CompareInputSerializer,
    ExplainInputSerializer,
    FeedbackSerializer,
    PredictInputSerializer,
    PredictionSerializer,
    SimilarQuerySerializer,
    TokenQuerySerializer,
)

logger = logging.getLogger(__name__)


def _save_prediction(user, payload):
    """Persist prediction history without failing the main inference response."""
    try:
        return Prediction.objects.create(
            user=user if user.is_authenticated else None,
            **payload,
        )
    except DatabaseError:
        logger.exception("Failed to persist prediction history")
        return None


def _temporary_inference_error(message='Inference service is temporarily unavailable. Please retry shortly.'):
    return Response({'detail': message}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


def _build_aspect_mentions(queryset, limit=8):
    """Aggregate aspect mentions from stored aspect arrays with review-text fallback."""
    counter = Counter()

    for aspects, review_text in queryset.values_list('aspects', 'review_text')[:1000]:
        review_aspects = set()

        if isinstance(aspects, list):
            for item in aspects:
                if isinstance(item, dict):
                    aspect_name = str(item.get('aspect', '')).strip().lower()
                else:
                    aspect_name = str(item).strip().lower()

                if aspect_name:
                    review_aspects.add(aspect_name)

        if not review_aspects and review_text:
            for token in ml_service._tokenize_words(review_text.lower()):
                normalized = re.sub(r'[^a-z]', '', token)
                if not normalized:
                    continue

                lemma = ml_service._safe_lemmatize(normalized)
                if normalized in MOVIE_ASPECTS:
                    review_aspects.add(normalized)
                elif lemma in MOVIE_ASPECTS:
                    review_aspects.add(lemma)

        for aspect_name in review_aspects:
            counter[aspect_name] += 1

    return [
        {'name': aspect_name, 'value': count}
        for aspect_name, count in counter.most_common(limit)
    ]


class PredictionPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class PredictView(APIView):
    """
    POST /api/predict/
    
    Predict sentiment of a movie review.
    
    Input: {"review": "This movie is amazing", "model": "logistic_regression|svm|bert|bert_vader"}
    Output: {"sentiment": "positive", "confidence": 0.94, ...}
    """
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = PredictInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        review = serializer.validated_data['review']
        model_name = serializer.validated_data.get('model', 'logistic_regression')
        try:
            result = ml_service.predict_with_model(review, model_name=model_name)
        except Exception:
            logger.exception("Prediction request failed for model=%s", model_name)
            return _temporary_inference_error()
        
        prediction = _save_prediction(request.user, {
            'review_text': review,
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'positive_prob': result['positive_prob'],
            'negative_prob': result['negative_prob'],
            'model_used': result.get('model_used', 'logistic_regression'),
        })
        
        return Response({
            'id': prediction.id if prediction else None,
            **result,
        })


class ExplainView(APIView):
    """
    POST /api/predict/explain/
    
    Predict sentiment with LIME word-level explanations.
    
    Input: {"review": "...", "num_features": 10}
    Output: {sentiment, confidence, explanation: [{word, weight, direction}], text_highlights: [...]}
    """
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = ExplainInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        review = serializer.validated_data['review']
        num_features = serializer.validated_data.get('num_features', 10)

        try:
            result = ml_service.explain(review, num_features=num_features)
        except Exception:
            logger.exception("Explain request failed")
            return _temporary_inference_error('Explainability service is temporarily unavailable. Please retry shortly.')
        
        prediction = _save_prediction(request.user, {
            'review_text': review,
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'positive_prob': result['positive_prob'],
            'negative_prob': result['negative_prob'],
            'model_used': 'logistic_regression',
            'explanation': result.get('explanation'),
        })
        
        return Response({
            'id': prediction.id if prediction else None,
            **result,
        })


class AspectView(APIView):
    """
    POST /api/predict/aspect/
    
    Aspect-based sentiment analysis.
    
    Input: {"review": "Acting was great but story was boring"}
    Output: {overall: {...}, aspects: [{aspect: "acting", sentiment: "positive"}, ...]}
    """
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        serializer = AspectInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        review = serializer.validated_data['review']
        try:
            result = ml_service.analyze_aspects(review)
        except Exception:
            logger.exception("Aspect request failed")
            return _temporary_inference_error('Aspect analysis is temporarily unavailable. Please retry shortly.')
        
        prediction = _save_prediction(request.user, {
            'review_text': review,
            'sentiment': result['overall']['sentiment'],
            'confidence': result['overall']['confidence'],
            'positive_prob': result['overall']['positive_prob'],
            'negative_prob': result['overall']['negative_prob'],
            'model_used': 'aspect_pipeline',
            'aspects': result.get('aspects'),
        })
        
        return Response({
            'id': prediction.id if prediction else None,
            **result,
        })


class PredictionHistoryView(generics.ListAPIView):
    """
    GET /api/predictions/
    
    Returns the authenticated user's prediction history.
    """
    serializer_class = PredictionSerializer
    permission_classes = [permissions.IsAuthenticated]
    pagination_class = PredictionPagination
    
    def get_queryset(self):
        queryset = Prediction.objects.filter(user=self.request.user)

        sentiment = self.request.query_params.get('sentiment')
        model_name = self.request.query_params.get('model')
        search_text = self.request.query_params.get('q')
        feedback = self.request.query_params.get('feedback')
        ordering = self.request.query_params.get('ordering', '-created_at')
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')

        if sentiment in {'positive', 'negative'}:
            queryset = queryset.filter(sentiment=sentiment)

        if model_name:
            queryset = queryset.filter(model_used=model_name)

        if search_text:
            queryset = queryset.filter(review_text__icontains=search_text)

        if feedback == 'corrected':
            queryset = queryset.exclude(user_correct__isnull=True).exclude(user_correct='')
        elif feedback == 'uncorrected':
            queryset = queryset.filter(Q(user_correct__isnull=True) | Q(user_correct=''))

        if start_date:
            start = parse_date(start_date)
            if start:
                queryset = queryset.filter(created_at__date__gte=start)

        if end_date:
            end = parse_date(end_date)
            if end:
                queryset = queryset.filter(created_at__date__lte=end)

        allowed_ordering = {'created_at', '-created_at', 'confidence', '-confidence'}
        if ordering in allowed_ordering:
            queryset = queryset.order_by(ordering)

        return queryset


class CompareView(APIView):
    """POST /api/predict/compare/ to compare LR, SVM, and BERT outputs."""

    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = CompareInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        review = serializer.validated_data['review']
        try:
            result = ml_service.compare_models(review)
        except Exception:
            logger.exception("Compare request failed")
            return _temporary_inference_error('Model comparison is temporarily unavailable. Please retry shortly.')
        winner = result['winner']
        winner_data = result['models'][winner]

        prediction = _save_prediction(request.user, {
            'review_text': review,
            'sentiment': winner_data['sentiment'],
            'confidence': winner_data['confidence'],
            'positive_prob': winner_data['positive_prob'],
            'negative_prob': winner_data['negative_prob'],
            'model_used': winner,
            'explanation': {'comparison': result['models']},
        })

        return Response({
            'id': prediction.id if prediction else None,
            **result,
        })


class BatchPredictView(APIView):
    """POST /api/predict/batch/ for 5-50 review predictions with summary."""

    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = BatchPredictInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        reviews = [item.strip() for item in serializer.validated_data['reviews'] if item.strip()]
        if len(reviews) < 5:
            return Response(
                {'detail': 'Provide at least 5 non-empty reviews for batch analysis.'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        results = []
        positive_count = 0
        negative_count = 0
        confidence_total = 0.0

        for index, review in enumerate(reviews):
            model_result = ml_service.predict(review)
            prediction = _save_prediction(request.user, {
                'review_text': review,
                'sentiment': model_result['sentiment'],
                'confidence': model_result['confidence'],
                'positive_prob': model_result['positive_prob'],
                'negative_prob': model_result['negative_prob'],
                'model_used': 'logistic_regression',
            })

            if model_result['sentiment'] == 'positive':
                positive_count += 1
            else:
                negative_count += 1

            confidence_total += model_result['confidence']
            results.append({
                'index': index,
                'id': prediction.id if prediction else None,
                'review': review,
                **model_result,
            })

        total = len(results)
        return Response({
            'count': total,
            'results': results,
            'summary': {
                'positive_count': positive_count,
                'negative_count': negative_count,
                'positive_percentage': round((positive_count / total) * 100, 1) if total else 0,
                'negative_percentage': round((negative_count / total) * 100, 1) if total else 0,
                'avg_confidence': round(confidence_total / total, 4) if total else 0,
            },
        })


class PredictionStatsView(APIView):
    """
    GET /api/predictions/stats/
    
    Returns statistics about the user's predictions.
    """
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        predictions = Prediction.objects.filter(user=request.user)
        total = predictions.count()
        positive = predictions.filter(sentiment='positive').count()
        negative = predictions.filter(sentiment='negative').count()
        avg_confidence = predictions.aggregate(avg=Avg('confidence'))['avg'] or 0
        aspect_mentions = _build_aspect_mentions(predictions, limit=8)

        recent_start = timezone.now() - timedelta(days=30)
        trend_queryset = (
            predictions
            .filter(created_at__gte=recent_start)
            .annotate(day=TruncDate('created_at'))
            .values('day')
            .annotate(
                total=Count('id'),
                positive=Count('id', filter=Q(sentiment='positive')),
                negative=Count('id', filter=Q(sentiment='negative')),
                avg_confidence=Avg('confidence'),
            )
            .order_by('day')
        )
        trend = [
            {
                'day': item['day'].isoformat(),
                'total': item['total'],
                'positive': item['positive'],
                'negative': item['negative'],
                'avg_confidence': round(item['avg_confidence'] or 0, 4),
            }
            for item in trend_queryset
        ]

        model_usage = list(
            predictions.values('model_used').annotate(count=Count('id')).order_by('-count')
        )
        
        return Response({
            'total_predictions': total,
            'positive_count': positive,
            'negative_count': negative,
            'positive_percentage': round(positive / total * 100, 1) if total else 0,
            'negative_percentage': round(negative / total * 100, 1) if total else 0,
            'avg_confidence': round(avg_confidence, 4),
            'trend': trend,
            'model_usage': model_usage,
            'aspect_mentions': aspect_mentions,
        })


class PredictionTokensView(APIView):
    """GET /api/predictions/tokens/ for word-cloud token frequency."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        serializer = TokenQuerySerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)

        limit = serializer.validated_data.get('limit', 30)
        sentiment = serializer.validated_data.get('sentiment')

        queryset = Prediction.objects.filter(user=request.user)
        if sentiment:
            queryset = queryset.filter(sentiment=sentiment)

        counter = Counter()
        for review in queryset.values_list('review_text', flat=True)[:1000]:
            for token in ml_service._tokenize_words(review.lower()):
                normalized = re.sub(r'[^a-z]', '', token)
                if len(normalized) < 3 or normalized in ml_service.stop_words:
                    continue
                counter[normalized] += 1

        tokens = [
            {'word': word, 'count': count}
            for word, count in counter.most_common(limit)
        ]

        return Response({
            'sentiment': sentiment or 'all',
            'tokens': tokens,
        })


class PredictionFeedbackView(APIView):
    """PATCH /api/predictions/<id>/feedback/ for correctness feedback."""

    permission_classes = [permissions.IsAuthenticated]

    def patch(self, request, prediction_id):
        prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
        serializer = FeedbackSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        updated_fields = []
        for field, value in serializer.validated_data.items():
            setattr(prediction, field, value)
            updated_fields.append(field)

        prediction.save(update_fields=updated_fields)
        return Response(PredictionSerializer(prediction).data)


class PredictionShareView(APIView):
    """POST /api/predictions/<id>/share/ to generate a public permalink."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, prediction_id):
        prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
        if not prediction.is_public:
            prediction.is_public = True
            prediction.save(update_fields=['is_public'])

        api_path = f'/api/predictions/shared/{prediction.share_uuid}/'
        frontend_path = f'/shared/{prediction.share_uuid}'

        return Response({
            'share_uuid': str(prediction.share_uuid),
            'api_url': request.build_absolute_uri(api_path),
            'frontend_path': frontend_path,
        })


class SharedPredictionDetailView(APIView):
    """GET /api/predictions/shared/<uuid>/ for public shared prediction view."""

    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get(self, request, share_uuid):
        prediction = (
            Prediction.objects
            .filter(share_uuid=share_uuid, is_public=True)
            .order_by('-created_at')
            .first()
        )
        if not prediction:
            raise Http404

        return Response({
            'id': prediction.id,
            'review_text': prediction.review_text,
            'sentiment': prediction.sentiment,
            'confidence': prediction.confidence,
            'positive_prob': prediction.positive_prob,
            'negative_prob': prediction.negative_prob,
            'model_used': prediction.model_used,
            'explanation': prediction.explanation,
            'aspects': prediction.aspects,
            'share_uuid': str(prediction.share_uuid),
            'created_at': prediction.created_at,
        })


class PredictionMetricsView(APIView):
    """GET /api/predictions/metrics/ for model card and calibration data."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        predictions = list(Prediction.objects.filter(user=request.user))
        total = len(predictions)

        if total == 0:
            empty_bins = [
                {
                    'bin': f'{i / 10:.1f}-{(i + 1) / 10:.1f}',
                    'count': 0,
                    'predicted': 0,
                    'observed': 0,
                }
                for i in range(10)
            ]
            return Response({
                'total_predictions': 0,
                'feedback_coverage': 0,
                'brier_score': 0,
                'calibration': empty_bins,
                'model_usage': [],
                'model_catalog': _model_catalog(),
            })

        feedback_count = sum(1 for item in predictions if item.user_correct)
        model_counts = Counter(item.model_used for item in predictions)

        calibration_bins = [
            {'count': 0, 'predicted_sum': 0.0, 'observed_sum': 0.0}
            for _ in range(10)
        ]
        brier_total = 0.0

        for item in predictions:
            predicted_positive = float(item.positive_prob)
            final_sentiment = item.user_correct or item.sentiment
            observed = 1.0 if final_sentiment == 'positive' else 0.0

            brier_total += (predicted_positive - observed) ** 2

            index = min(int(predicted_positive * 10), 9)
            bucket = calibration_bins[index]
            bucket['count'] += 1
            bucket['predicted_sum'] += predicted_positive
            bucket['observed_sum'] += observed

        calibration = []
        for index, bucket in enumerate(calibration_bins):
            count = bucket['count']
            calibration.append({
                'bin': f'{index / 10:.1f}-{(index + 1) / 10:.1f}',
                'count': count,
                'predicted': round(bucket['predicted_sum'] / count, 4) if count else 0,
                'observed': round(bucket['observed_sum'] / count, 4) if count else 0,
            })

        model_usage = [
            {'model_used': model_name, 'count': count}
            for model_name, count in model_counts.most_common()
        ]

        return Response({
            'total_predictions': total,
            'feedback_coverage': round(feedback_count / total, 4),
            'brier_score': round(brier_total / total, 4),
            'calibration': calibration,
            'model_usage': model_usage,
            'model_catalog': _model_catalog(),
        })


def _model_catalog():
    """Static model metadata for dashboard model-card rendering."""
    return [
        {
            'id': 'logistic_regression',
            'name': 'TF-IDF + Logistic Regression',
            'family': 'Classical ML',
            'status': 'active' if ml_service.model_loaded else 'fallback-demo',
            'reported_accuracy': 0.902,
        },
        {
            'id': 'svm',
            'name': 'Linear SVM (Calibrated)',
            'family': 'Classical ML',
            'status': 'active' if ml_service.svm_model_loaded else 'not-loaded',
            'reported_accuracy': 0.91,
        },
        {
            'id': 'bert',
            'name': 'BERT Fine-Tuned',
            'family': 'Transformer',
            'status': 'active' if (ml_service.bert_model_loaded or ml_service.remote_bert_enabled) else 'runtime-optional',
            'reported_accuracy': 0.86,
        },
        {
            'id': 'bert_vader',
            'name': 'BERT + VADER Fusion',
            'family': 'Hybrid Ensemble',
            'status': 'active' if ml_service.vader_analyzer else 'runtime-optional',
            'reported_accuracy': 0.89,
        },
    ]


class SimilarPredictionView(APIView):
    """GET /api/predictions/similar/ to search nearest historical reviews."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        serializer = SimilarQuerySerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)

        review = serializer.validated_data['review']
        limit = serializer.validated_data.get('limit', 5)

        candidates = list(
            Prediction.objects.filter(user=request.user)
            .order_by('-created_at')[:400]
        )

        if not candidates:
            return Response({'query': review, 'results': []})

        texts = [candidate.review_text for candidate in candidates]
        scores = ml_service.similarity_scores(review, texts)

        results = []
        for candidate, score in zip(candidates, scores):
            if candidate.review_text.strip().lower() == review.strip().lower():
                continue
            results.append({
                'id': candidate.id,
                'review_text': candidate.review_text,
                'sentiment': candidate.sentiment,
                'confidence': candidate.confidence,
                'similarity': round(score, 4),
                'created_at': candidate.created_at,
            })

        results.sort(key=lambda item: item['similarity'], reverse=True)
        return Response({
            'query': review,
            'results': results[:limit],
        })


class HealthCheckView(APIView):
    """GET /api/health/ for Render uptime checks and external monitoring."""

    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    throttle_classes = []

    def get(self, request):
        resp = Response({
            'status': 'ok',
            'service': 'cinescope-api',
            'timestamp': timezone.now().isoformat(),
            'model_mode': 'trained' if ml_service.model_loaded else 'lazy',
            'models_loaded': bool(
                ml_service.model_loaded or ml_service.svm_model_loaded or ml_service.bert_model_loaded
            ),
        }, status=status.HTTP_200_OK)
        resp['Cache-Control'] = 'public, max-age=10'
        return resp
