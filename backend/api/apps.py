from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        """
        Pre-warm the ML service and NLTK resources at startup so the first
        real request is not slow.  This runs once per worker process.
        """
        import os
        # Skip during Django management commands (migrate, collectstatic, etc.)
        if os.environ.get('RUN_MAIN') == 'true' or not os.environ.get('DJANGO_SETTINGS_MODULE'):
            return

        try:
            from .ml_service import ml_service
            # Trigger lazy model load and NLP init at startup, not on first request
            ml_service._ensure_models_loaded()
            ml_service._ensure_nlp_loaded()
        except Exception:
            pass  # Never block startup for ML warm-up failures
