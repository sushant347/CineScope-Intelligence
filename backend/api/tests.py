from rest_framework import status
from rest_framework.test import APITestCase

from accounts.models import User
from api.models import Prediction


class APISmokeTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='tester',
            email='tester@example.com',
            password='StrongPass123!'
        )

    def test_health_endpoint(self):
        response = self.client.get('/api/health/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('status', response.data)

    def test_predict_endpoint_returns_payload(self):
        response = self.client.post(
            '/api/predict/',
            {
                'review': 'Strong performances and sharp writing made this film memorable.',
                'model': 'svm',
            },
            format='json',
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('sentiment', response.data)
        self.assertIn('confidence', response.data)
        self.assertIn('positive_prob', response.data)
        self.assertIn('negative_prob', response.data)
        self.assertIn('model_used', response.data)
        self.assertIn('model_requested', response.data)

    def test_compare_endpoint_returns_model_map(self):
        response = self.client.post(
            '/api/predict/compare/',
            {'review': 'A tight screenplay with sharp performances and a gripping ending.'},
            format='json',
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('winner', response.data)
        self.assertIn('models', response.data)
        self.assertIn('logistic_regression', response.data['models'])

    def test_batch_endpoint_accepts_multiple_reviews(self):
        payload = {
            'reviews': [
                'Great pacing and strong emotional payoff.',
                'Weak character arc but stylish visuals.',
                'Excellent direction and memorable score.',
                'The second act drags and the ending feels rushed.',
                'Solid performances made this worth watching.',
            ]
        }
        response = self.client.post('/api/predict/batch/', payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['count'], 5)
        self.assertIn('summary', response.data)

    def test_authenticated_history_extensions(self):
        prediction = Prediction.objects.create(
            user=self.user,
            review_text='Thoughtful writing and good performances.',
            sentiment='positive',
            confidence=0.82,
            positive_prob=0.82,
            negative_prob=0.18,
            model_used='logistic_regression',
        )

        self.client.force_authenticate(user=self.user)

        history_response = self.client.get('/api/predictions/?q=thoughtful&sentiment=positive')
        self.assertEqual(history_response.status_code, status.HTTP_200_OK)

        feedback_response = self.client.patch(
            f'/api/predictions/{prediction.id}/feedback/',
            {'user_correct': 'positive', 'is_public': True},
            format='json',
        )
        self.assertEqual(feedback_response.status_code, status.HTTP_200_OK)
        self.assertEqual(feedback_response.data['user_correct'], 'positive')

        share_response = self.client.post(f'/api/predictions/{prediction.id}/share/', {}, format='json')
        self.assertEqual(share_response.status_code, status.HTTP_200_OK)
        self.assertIn('share_uuid', share_response.data)

        shared_response = self.client.get(f"/api/predictions/shared/{share_response.data['share_uuid']}/")
        self.assertEqual(shared_response.status_code, status.HTTP_200_OK)
        self.assertEqual(shared_response.data['sentiment'], 'positive')

        stats_response = self.client.get('/api/predictions/stats/')
        self.assertEqual(stats_response.status_code, status.HTTP_200_OK)
        self.assertIn('trend', stats_response.data)

        tokens_response = self.client.get('/api/predictions/tokens/?limit=10')
        self.assertEqual(tokens_response.status_code, status.HTTP_200_OK)
        self.assertIn('tokens', tokens_response.data)

        metrics_response = self.client.get('/api/predictions/metrics/')
        self.assertEqual(metrics_response.status_code, status.HTTP_200_OK)
        self.assertIn('calibration', metrics_response.data)

        similar_response = self.client.get(
            '/api/predictions/similar/?review=Thoughtful writing and good performances.&limit=3'
        )
        self.assertEqual(similar_response.status_code, status.HTTP_200_OK)
        self.assertIn('results', similar_response.data)
