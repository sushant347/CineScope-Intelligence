from transformers import pipeline

p = pipeline('text-classification', model='../ml/models/bert_sentiment')
print('Prediction for "Good":', p('Good'))
print('Prediction for "Best movie ever":', p('Best movie ever'))
print('Prediction for "A sharp screenplay, excellent cast chemistry, and an ending that actually earns its emotion.":', p('A sharp screenplay, excellent cast chemistry, and an ending that actually earns its emotion.'))
