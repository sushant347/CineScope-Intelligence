"""API Serializers"""

from rest_framework import serializers
from .models import Prediction

PREDICT_MODEL_CHOICES = ['logistic_regression', 'svm', 'bert']


class PredictInputSerializer(serializers.Serializer):
    review = serializers.CharField(min_length=5, max_length=10000)
    model = serializers.ChoiceField(
        required=False,
        choices=PREDICT_MODEL_CHOICES,
        default='logistic_regression',
    )


class CompareInputSerializer(serializers.Serializer):
    review = serializers.CharField(min_length=5, max_length=10000)


class BatchPredictInputSerializer(serializers.Serializer):
    reviews = serializers.ListField(
        child=serializers.CharField(min_length=5, max_length=10000),
        min_length=5,
        max_length=50,
    )


class FeedbackSerializer(serializers.Serializer):
    user_correct = serializers.ChoiceField(
        choices=[choice[0] for choice in Prediction.FEEDBACK_CHOICES],
        required=False,
        allow_null=True,
    )
    feedback_note = serializers.CharField(required=False, allow_blank=True, max_length=300)
    is_public = serializers.BooleanField(required=False)

    def validate(self, attrs):
        if not attrs:
            raise serializers.ValidationError('At least one field must be provided.')
        return attrs


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = (
            'id', 'review_text', 'sentiment', 'confidence',
            'positive_prob', 'negative_prob', 'model_used',
            'explanation', 'aspects', 'user_correct', 'feedback_note',
            'is_public', 'share_uuid', 'created_at'
        )
        read_only_fields = fields


class ExplainInputSerializer(serializers.Serializer):
    review = serializers.CharField(min_length=5, max_length=10000)
    num_features = serializers.IntegerField(default=10, min_value=5, max_value=20)


class AspectInputSerializer(serializers.Serializer):
    review = serializers.CharField(min_length=10, max_length=10000)


class SimilarQuerySerializer(serializers.Serializer):
    review = serializers.CharField(min_length=5, max_length=10000)
    limit = serializers.IntegerField(required=False, min_value=1, max_value=10, default=5)


class TokenQuerySerializer(serializers.Serializer):
    limit = serializers.IntegerField(required=False, min_value=5, max_value=60, default=30)
    sentiment = serializers.ChoiceField(
        required=False,
        allow_blank=True,
        choices=['positive', 'negative'],
    )
