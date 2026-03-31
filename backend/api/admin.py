from django.contrib import admin
from .models import Prediction

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'sentiment', 'confidence', 'model_used',
        'user_correct', 'is_public', 'created_at', 'short_review'
    )
    list_filter = ('sentiment', 'model_used', 'user_correct', 'is_public', 'created_at')
    search_fields = ('review_text',)
    readonly_fields = ('created_at', 'share_uuid')
    
    def short_review(self, obj):
        return obj.review_text[:80] + '...' if len(obj.review_text) > 80 else obj.review_text
    short_review.short_description = 'Review'
