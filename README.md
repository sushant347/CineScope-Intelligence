# CineScope Intelligence

Full-stack movie review sentiment analysis system with multi-model inference, explainability, analytics, and deployment support.

## Core Features

- Standard sentiment prediction with confidence scores
- Runtime model selection in Analyzer (Logistic Regression, SVM, BERT with graceful fallback)
- Explainable prediction mode with LIME highlights
- Aspect-level sentiment analysis for film-specific entities
- Batch analysis (5 to 50 reviews) with CSV import/export
- Comparison mode across LR/SVM/BERT
- User history with filtering, pagination, and charts
- Feedback loop (user-corrected labels)
- Shareable public prediction links
- Similar review search over saved history
- Metrics page with calibration and model usage

## Tech Stack

- ML: scikit-learn, NLTK, PyTorch, Transformers, LIME
- Backend: Django, Django REST Framework, SimpleJWT
- Frontend: React + Vite + Recharts + Framer Motion
- Deployment: Neon PostgreSQL + Render

## Repository Layout

- backend/: Django API, auth, persistence
- frontend/: React web client
- ml/: training scripts, preprocessing pipeline, model artifacts
- render.yaml: Render Blueprint configuration
- deployment.md: deployment instructions
- project.md: complete feature and architecture notes

## Local Setup

### 1. Create backend-local virtual environment

```bash
cd backend
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
cd ../ml
../backend/.venv/Scripts/python.exe -m pip install -r requirements.txt
cd ../frontend
npm install
cd ..
```

### 2. Run migrations

```bash
cd backend
.venv/Scripts/python.exe manage.py migrate
cd ..
```

### 3. Train models

```bash
cd ml
../backend/.venv/Scripts/python.exe train.py --phase preprocess
../backend/.venv/Scripts/python.exe train.py --phase baseline
../backend/.venv/Scripts/python.exe train.py --phase advanced
../backend/.venv/Scripts/python.exe train.py --phase bert --batch-size 8
cd ..
```

Notes:
- CPU-only systems automatically use a reduced BERT training profile.
- Baseline and advanced artifacts are written under ml/models.

### 4. Run services

```bash
# Terminal 1
cd backend
.venv/Scripts/python.exe manage.py runserver

# Terminal 2
cd frontend
npm run dev
```

- Backend URL: http://127.0.0.1:8000
- Frontend URL: http://localhost:5173

## How To Use Different Models

1. Open Analyzer.
2. Keep mode as Standard Inference.
3. Select model from the new Choose prediction model dropdown.
4. Run analysis.

If the selected model is unavailable (for example missing BERT runtime), the API falls back safely and returns the actual model used plus reason.

## API Endpoints

### Inference

- GET /api/health/
- POST /api/predict/
- POST /api/predict/compare/
- POST /api/predict/batch/
- POST /api/predict/explain/
- POST /api/predict/aspect/

### Prediction History and Analytics (authenticated)

- GET /api/predictions/
- GET /api/predictions/stats/
- GET /api/predictions/tokens/
- GET /api/predictions/metrics/
- GET /api/predictions/similar/
- PATCH /api/predictions/<id>/feedback/
- POST /api/predictions/<id>/share/
- GET /api/predictions/shared/<uuid>/

### Authentication

- POST /api/auth/register/
- POST /api/auth/login/
- POST /api/auth/token/refresh/
- GET/PUT/PATCH /api/auth/profile/

## Configuration Notes

- Backend env template: backend/.env.example
- Frontend env template: frontend/.env.example
- DATABASE_URL enables Neon/Postgres usage in production
- ML_MODEL_DIR controls model artifact path

## Deployment

Use deployment.md for full Neon + Render instructions.

## Quality Checks

```bash
# Backend tests
cd backend
.venv/Scripts/python.exe manage.py test

# Frontend build
cd ../frontend
npm run build
```
