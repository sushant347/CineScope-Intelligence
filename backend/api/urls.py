"""API URLs"""

from django.urls import path
from .views import (
    AspectView,
    BatchPredictView,
    CompareView,
    ExplainView,
    HealthCheckView,
    PredictView,
    PredictionFeedbackView,
    PredictionHistoryView,
    PredictionMetricsView,
    PredictionShareView,
    PredictionStatsView,
    PredictionTokensView,
    SharedPredictionDetailView,
    SimilarPredictionView,
)

urlpatterns = [
    path('health/', HealthCheckView.as_view(), name='health'),
    path('predict/', PredictView.as_view(), name='predict'),
    path('predict/compare/', CompareView.as_view(), name='predict-compare'),
    path('predict/batch/', BatchPredictView.as_view(), name='predict-batch'),
    path('predict/explain/', ExplainView.as_view(), name='explain'),
    path('predict/aspect/', AspectView.as_view(), name='aspect'),
    path('predictions/', PredictionHistoryView.as_view(), name='predictions'),
    path('predictions/stats/', PredictionStatsView.as_view(), name='prediction-stats'),
    path('predictions/tokens/', PredictionTokensView.as_view(), name='prediction-tokens'),
    path('predictions/metrics/', PredictionMetricsView.as_view(), name='prediction-metrics'),
    path('predictions/similar/', SimilarPredictionView.as_view(), name='prediction-similar'),
    path('predictions/<int:prediction_id>/feedback/', PredictionFeedbackView.as_view(), name='prediction-feedback'),
    path('predictions/<int:prediction_id>/share/', PredictionShareView.as_view(), name='prediction-share'),
    path('predictions/shared/<uuid:share_uuid>/', SharedPredictionDetailView.as_view(), name='prediction-shared-detail'),
]
