import axios from 'axios';

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');
const API_TIMEOUT_MS = Number(import.meta.env.VITE_API_TIMEOUT_MS || 35000);
const RETRYABLE_STATUS = new Set([408, 429, 500, 502, 503, 504]);

const api = axios.create({
  baseURL: API_BASE,
  timeout: Number.isFinite(API_TIMEOUT_MS) ? API_TIMEOUT_MS : 35000,
  headers: { 'Content-Type': 'application/json' },
});

const sleep = (delayMs) => new Promise((resolve) => window.setTimeout(resolve, delayMs));

const isTransientRequestError = (error) => {
  if (error?.code === 'ECONNABORTED') {
    return true;
  }
  if (!error?.response) {
    return true;
  }
  return RETRYABLE_STATUS.has(error.response.status);
};

const withRetry = async (requestFn, { retries = 1, delayMs = 1200 } = {}) => {
  let attempt = 0;
  let lastError;

  while (attempt <= retries) {
    try {
      return await requestFn();
    } catch (error) {
      lastError = error;
      if (attempt >= retries || !isTransientRequestError(error)) {
        throw error;
      }
      await sleep(delayMs * (attempt + 1));
      attempt += 1;
    }
  }

  throw lastError;
};

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
  withRetry(() => api.post('/predict/', { review, model }), { retries: 1, delayMs: 1500 });

export const comparePredictions = (review) =>
  withRetry(() => api.post('/predict/compare/', { review }), { retries: 1, delayMs: 1500 });

export const predictBatch = (reviews) =>
  withRetry(() => api.post('/predict/batch/', { reviews }), { retries: 1, delayMs: 1500 });

export const predictWithExplanation = (review, numFeatures = 10) =>
  withRetry(
    () => api.post('/predict/explain/', { review, num_features: numFeatures }),
    { retries: 1, delayMs: 1500 }
  );

export const predictAspects = (review) =>
  withRetry(() => api.post('/predict/aspect/', { review }), { retries: 1, delayMs: 1500 });

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
