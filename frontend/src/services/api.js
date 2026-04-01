import axios from 'axios';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

const toQueryString = (params = {}) => {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      searchParams.append(key, String(value));
    }
  });
  return searchParams.toString();
};

// Attach JWT token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle token refresh
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    if (error.response?.status === 401 && !originalRequest?._retry) {
      originalRequest._retry = true;
      const refreshToken = localStorage.getItem('refresh_token');
      if (refreshToken) {
        try {
          const { data } = await axios.post(`${API_BASE}/auth/token/refresh/`, {
            refresh: refreshToken,
          });
          localStorage.setItem('access_token', data.access);
          originalRequest.headers.Authorization = `Bearer ${data.access}`;
          return api(originalRequest);
        } catch {
          localStorage.removeItem('access_token');
          localStorage.removeItem('refresh_token');
          localStorage.removeItem('username');
        }
      }
    }
    return Promise.reject(error);
  }
);

// Auth
export const login = (username, password) =>
  api.post('/auth/login/', { username, password });

export const register = (username, email, password, password2) =>
  api.post('/auth/register/', { username, email, password, password2 });

export const getProfile = () => api.get('/auth/profile/');

// Predictions
export const predictSentiment = (review, model = 'logistic_regression') =>
  api.post('/predict/', { review, model });

export const comparePredictions = (review) =>
  api.post('/predict/compare/', { review });

export const predictBatch = (reviews) =>
  api.post('/predict/batch/', { reviews });

export const predictWithExplanation = (review, numFeatures = 10) =>
  api.post('/predict/explain/', { review, num_features: numFeatures });

export const predictAspects = (review) =>
  api.post('/predict/aspect/', { review });

// History
export const getPredictions = (params = {}) =>
  api.get(`/predictions/?${toQueryString(params)}`);

export const getPredictionStats = () =>
  api.get('/predictions/stats/');

export const getPredictionTokens = (params = {}) =>
  api.get(`/predictions/tokens/?${toQueryString(params)}`);

export const updatePredictionFeedback = (predictionId, payload) =>
  api.patch(`/predictions/${predictionId}/feedback/`, payload);

export const createPredictionShare = (predictionId) =>
  api.post(`/predictions/${predictionId}/share/`, {});

export const getSharedPrediction = (shareUuid) =>
  api.get(`/predictions/shared/${shareUuid}/`);

export const getPredictionMetrics = () =>
  api.get('/predictions/metrics/');

export const searchSimilarPredictions = (review, limit = 5) =>
  api.get(`/predictions/similar/?${toQueryString({ review, limit })}`);

export default api;
