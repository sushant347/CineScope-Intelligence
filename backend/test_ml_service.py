import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from api.ml_service import ml_service
print("Local BERT model loaded?", ml_service.bert_model_loaded)

res_pos = ml_service.predict_with_model('I love this movie!', model_name='bert')
print("Output for 'I love this movie!':", res_pos)

res_neg = ml_service.predict_with_model('I hate this terrible movie.', model_name='bert')
print("Output for 'I hate this terrible movie.':", res_neg)
