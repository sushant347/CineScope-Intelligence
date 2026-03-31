"""API Models - Prediction History"""

from django.db import models
from django.conf import settings
import uuid


class Prediction(models.Model):
    """Stores individual sentiment predictions."""
    FEEDBACK_CHOICES = (
        ('positive', 'Positive'),
        ('negative', 'Negative'),
    )

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='predictions',
        null=True, blank=True
    )
    review_text = models.TextField()
    sentiment = models.CharField(max_length=10)
    confidence = models.FloatField()
    positive_prob = models.FloatField()
    negative_prob = models.FloatField()
    model_used = models.CharField(max_length=50, default='logistic_regression')
    explanation = models.JSONField(null=True, blank=True)
    aspects = models.JSONField(null=True, blank=True)
    user_correct = models.CharField(max_length=10, choices=FEEDBACK_CHOICES, null=True, blank=True)
    feedback_note = models.CharField(max_length=300, blank=True)
    is_public = models.BooleanField(default=False)
    share_uuid = models.UUIDField(default=uuid.uuid4, editable=False, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.sentiment} ({self.confidence:.2f}) - {self.review_text[:50]}"
