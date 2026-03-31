"""URL Configuration"""

from django.contrib import admin
from django.urls import path, include
from rest_framework.permissions import AllowAny
from rest_framework.schemas import get_schema_view

schema_view = get_schema_view(
    title='CineScope Intelligence API',
    description='Sentiment analysis API with explainability and aspect endpoints.',
    version='1.0.0',
    public=True,
    permission_classes=[AllowAny],
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/schema/', schema_view, name='api-schema'),
    path('api/', include('api.urls')),
    path('api/auth/', include('accounts.urls')),
]
