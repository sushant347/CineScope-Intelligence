from rest_framework import status
from rest_framework.test import APITestCase


class AccountFlowTests(APITestCase):
    def test_register_and_login(self):
        register_payload = {
            'username': 'test_user',
            'email': 'test_user@example.com',
            'password': 'secure1234',
            'password2': 'secure1234',
        }

        register_response = self.client.post('/api/auth/register/', register_payload, format='json')
        self.assertEqual(register_response.status_code, status.HTTP_201_CREATED)

        login_response = self.client.post(
            '/api/auth/login/',
            {'username': 'test_user', 'password': 'secure1234'},
            format='json',
        )
        self.assertEqual(login_response.status_code, status.HTTP_200_OK)
        self.assertIn('access', login_response.data)
        self.assertIn('refresh', login_response.data)
