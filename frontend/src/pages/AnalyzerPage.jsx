import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  FiActivity,
  FiCheckCircle,
  FiDownload,
  FiInfo,
  FiLayers,
  FiRepeat,
  FiSearch,
  FiUpload,
} from 'react-icons/fi';
import {
  comparePredictions,
  predictAspects,
  predictBatch,
  predictSentiment,
  predictWithExplanation,
  searchSimilarPredictions,
} from '../services/api';

const SAMPLE_REVIEWS = [
  'A sharp screenplay, excellent cast chemistry, and an ending that actually earns its emotion.',
  'Great visuals, weak pacing. The soundtrack carries the film but the story feels rushed.',
  'Overlong and inconsistent. A few strong performances cannot rescue the clumsy writing.',
];

const MODES = [
  {
    id: 'basic',
    label: 'Standard Inference',
    icon: FiActivity,
    helper: 'Fast polarity and confidence for quick validation.',
  },
  {
    id: 'explain',
    label: 'Explainable Inference',
    icon: FiSearch,
    helper: 'Adds word-level attribution for interpretability.',
  },
  {
    id: 'aspect',
    label: 'Aspect Breakdown',
    icon: FiLayers,
    helper: 'Finds sentence-level sentiment for domain aspects.',
  },
  {
    id: 'compare',
    label: 'Model Compare',
    icon: FiRepeat,
    helper: 'Compares LR, SVM, and BERT predictions side-by-side.',
  },
];

const formatPercent = (value) => `${((value || 0) * 100).toFixed(1)}%`;

const normalizeSentiment = (value) => {
  const sentiment = String(value || '').toLowerCase();
  if (['positive', 'negative', 'mixed', 'neutral'].includes(sentiment)) {
    return sentiment;
  }
  return 'neutral';
};

const statusClassName = (value) => `status-${normalizeSentiment(value)}`;

const MODEL_LABELS = {
  logistic_regression: 'Logistic Regression',
  svm: 'SVM',
  bert: 'BERT',
  bert_vader: 'BERT + VADER Fusion',
};

const INFERENCE_MODELS = [
  { id: 'logistic_regression', label: 'Logistic Regression (fast default)' },
  { id: 'svm', label: 'SVM (classical alternative)' },
  { id: 'bert', label: 'BERT (best contextual model when available)' },
  { id: 'bert_vader', label: 'BERT + VADER fusion (pos/neg/mixed + score + keywords)' },
];

const parseCsvLine = (line) => {
  const trimmed = line.trim();
  if (!trimmed) {
    return '';
  }

  const quotedMatch = trimmed.match(/^"((?:[^"]|"")*)"(?:,.*)?$/);
  if (quotedMatch) {
    return quotedMatch[1].replace(/""/g, '"').trim();
  }

  const commaIndex = trimmed.indexOf(',');
  return (commaIndex === -1 ? trimmed : trimmed.slice(0, commaIndex)).trim();
};

const toReviewList = (rawText) =>
  rawText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length >= 5);

const toCsv = (rows) => {
  const header = ['index', 'sentiment', 'confidence', 'positive_prob', 'negative_prob', 'review'];
  const body = rows.map((row) => {
    const escapedReview = `"${(row.review || '').replace(/"/g, '""')}"`;
    return [
      row.index,
      row.sentiment,
      row.confidence,
      row.positive_prob,
      row.negative_prob,
      escapedReview,
    ].join(',');
  });

  return [header.join(','), ...body].join('\n');
};

const getAnalyzeErrorMessage = (err) => {
  const detail = err?.response?.data?.detail || err?.response?.data?.message || err?.response?.data?.reason;
  if (detail) {
    return detail;
  }

  if (err?.code === 'ECONNABORTED') {
    return 'Analysis timed out while the server was processing. Please retry in a few seconds.';
  }

  const status = err?.response?.status;
  if (status === 502 || status === 503 || status === 504) {
    return 'Backend is waking up or temporarily overloaded. Please retry in a few seconds.';
  }

  if (!err?.response) {
    return 'Network issue while contacting backend. Check connection and retry.';
  }

  return 'Analysis failed. Please retry.';
};

function AnalyzerPage() {
  const [review, setReview] = useState('');
  const [selectedMode, setSelectedMode] = useState('basic');
  const [selectedModel, setSelectedModel] = useState('logistic_regression');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [similarReviews, setSimilarReviews] = useState([]);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [similarError, setSimilarError] = useState('');

  const [batchText, setBatchText] = useState('');
  const [batchResult, setBatchResult] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState('');
  const [batchProgress, setBatchProgress] = useState(0);

  const activeMode = useMemo(
    () => MODES.find((mode) => mode.id === selectedMode) || MODES[0],
    [selectedMode]
  );

  const sentiment = result?.sentiment || result?.overall?.sentiment || 'neutral';
  const sentimentTone = normalizeSentiment(sentiment);
  const confidence = result?.confidence ?? result?.overall?.confidence ?? 0;
  const positiveProb = result?.positive_prob ?? result?.overall?.positive_prob ?? 0;
  const negativeProb = result?.negative_prob ?? result?.overall?.negative_prob ?? 0;
  const batchReviews = useMemo(() => toReviewList(batchText), [batchText]);

  const handleAnalyze = async () => {
    if (!review.trim() || review.trim().length < 5) {
      setError('Please enter at least 5 characters to analyze.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);
    setSimilarReviews([]);
    setSimilarError('');

    try {
      let response;
      if (selectedMode === 'explain') {
        response = await predictWithExplanation(review);
      } else if (selectedMode === 'aspect') {
        response = await predictAspects(review);
      } else if (selectedMode === 'compare') {
        response = await comparePredictions(review);
      } else {
        response = await predictSentiment(review, selectedModel);
      }

      setResult({ ...response.data, mode: selectedMode });
    } catch (err) {
      setError(getAnalyzeErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleFindSimilar = async () => {
    if (!review.trim() || review.trim().length < 5) {
      setSimilarError('Enter a review first to run similarity search.');
      return;
    }

    setSimilarLoading(true);
    setSimilarError('');
    setSimilarReviews([]);

    try {
      const response = await searchSimilarPredictions(review, 5);
      setSimilarReviews(response.data.results || []);
    } catch (err) {
      if (err.response?.status === 401) {
        setSimilarError('Login is required to search similar reviews in your saved history.');
      } else {
        setSimilarError('Unable to search similar reviews right now.');
      }
    } finally {
      setSimilarLoading(false);
    }
  };

  const handleBatchAnalyze = async () => {
    if (batchReviews.length < 5 || batchReviews.length > 50) {
      setBatchError('Batch mode requires between 5 and 50 reviews.');
      return;
    }

    setBatchLoading(true);
    setBatchError('');
    setBatchResult(null);
    setBatchProgress(8);

    const timer = setInterval(() => {
      setBatchProgress((previous) => (previous >= 92 ? previous : previous + 7));
    }, 260);

    try {
      const response = await predictBatch(batchReviews);
      setBatchResult(response.data);
      setBatchProgress(100);
    } catch (err) {
      setBatchError(err.response?.data?.detail || 'Batch analysis failed. Check your input and try again.');
      setBatchProgress(0);
    } finally {
      clearInterval(timer);
      setBatchLoading(false);
      window.setTimeout(() => setBatchProgress(0), 700);
    }
  };

  const handleBatchFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    try {
      const content = await file.text();
      const lines = content.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
      const normalized = lines.map(parseCsvLine).filter((line) => line.length >= 5);

      const withoutHeader = normalized[0]?.toLowerCase() === 'review'
        ? normalized.slice(1)
        : normalized;

      setBatchText(withoutHeader.join('\n'));
      setBatchError('');
    } catch {
      setBatchError('Could not read the selected file.');
    }
  };

  const handleBatchExport = () => {
    const rows = batchResult?.results || [];
    if (!rows.length) {
      return;
    }

    const csv = toCsv(rows);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `batch-predictions-${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <motion.div
      className="analyzer-page"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
    >
      <div className="container analyzer-shell">
        <section className="panel panel-input">
          <div className="panel-heading">
            <h1>Review Analyzer</h1>
            <p>Paste a review and run one of four inference pipelines.</p>
          </div>

          <textarea
            className="review-input"
            value={review}
            onChange={(event) => setReview(event.target.value)}
            placeholder="Example: The directing is precise and the performances are natural, but the pacing in the second act slows everything down."
            maxLength={5000}
          />

          <div className="input-meta">
            <span>{review.length} / 5000</span>
            <span>Minimum: 5 characters</span>
          </div>

          <div className="sample-row">
            {SAMPLE_REVIEWS.map((sample) => (
              <button
                key={sample}
                type="button"
                className="sample-chip"
                onClick={() => {
                  setReview(sample);
                  setError('');
                }}
              >
                {sample}
              </button>
            ))}
          </div>

          <div className="mode-grid">
            {MODES.map((mode) => {
              const Icon = mode.icon;
              return (
                <button
                  key={mode.id}
                  type="button"
                  className={`mode-card ${selectedMode === mode.id ? 'active' : ''}`}
                  onClick={() => setSelectedMode(mode.id)}
                >
                  <div>
                    <Icon />
                    <strong>{mode.label}</strong>
                  </div>
                  <p>{mode.helper}</p>
                </button>
              );
            })}
          </div>

          {selectedMode === 'basic' && (
            <div className="model-select-wrap">
              <label htmlFor="model-select">Choose prediction model</label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
              >
                {INFERENCE_MODELS.map((model) => (
                  <option key={model.id} value={model.id}>{model.label}</option>
                ))}
              </select>
              <small>
                If a selected model is unavailable at runtime, the API will safely fall back and report the used model.
              </small>
            </div>
          )}

          <button type="button" className="btn btn-primary analyze-btn" onClick={handleAnalyze} disabled={loading}>
            <FiActivity />
            {loading ? 'Running Analysis...' : `Run ${activeMode.label}`}
          </button>
        </section>

        <section className="panel panel-result">
          {error && (
            <div className="status-banner error">
              <FiInfo />
              <span>{error}</span>
            </div>
          )}

          {loading && (
            <div className="result-empty">
              <div className="spinner"></div>
              <p>Processing request and scoring sentiment...</p>
            </div>
          )}

          {!loading && !result && !error && (
            <div className="result-empty">
              <FiCheckCircle />
              <p>Your inference output will appear here.</p>
            </div>
          )}

          {!loading && result && (
            <div className="result-stack">
              <article className="result-card">
                <div className="result-header">
                  <div>
                    <h2>Prediction Summary</h2>
                    <p>{activeMode.helper}</p>
                  </div>
                  <span className={`sentiment-pill ${sentimentTone}`}>{sentiment}</span>
                </div>

                {result.model_used && (
                  <p className="helper-text">
                    Model used: {MODEL_LABELS[result.model_used] || result.model_used}
                    {result.model_requested && result.model_requested !== result.model_used
                      ? ` (requested ${MODEL_LABELS[result.model_requested] || result.model_requested})`
                      : ''}
                  </p>
                )}

                {result.fallback && result.reason && (
                  <div className="status-banner">
                    <span>{result.reason}</span>
                  </div>
                )}

                <div className="confidence-row">
                  <div>
                    <span>Confidence</span>
                    <strong>{formatPercent(confidence)}</strong>
                  </div>
                  <div className="progress-track">
                    <div
                      className={`progress-fill ${sentimentTone}`}
                      style={{ width: `${Math.max(confidence * 100, 3)}%` }}
                    />
                  </div>
                </div>

                <div className="probability-grid">
                  <div className="probability-card positive">
                    <span>Positive</span>
                    <strong>{formatPercent(positiveProb)}</strong>
                  </div>
                  <div className="probability-card negative">
                    <span>Negative</span>
                    <strong>{formatPercent(negativeProb)}</strong>
                  </div>
                </div>

                {typeof result.score === 'number' && (
                  <div className="fusion-score-row">
                    <span>Composite Score</span>
                    <strong>{result.score.toFixed(2)}</strong>
                  </div>
                )}

                {Array.isArray(result.keywords) && result.keywords.length > 0 && (
                  <div className="keyword-row">
                    <span>Keywords</span>
                    <div className="keyword-chips">
                      {result.keywords.map((keyword) => (
                        <span key={keyword} className="keyword-chip">{keyword}</span>
                      ))}
                    </div>
                  </div>
                )}
              </article>

              {result.mode === 'explain' && (
                <article className="result-card">
                  <h3>Word-Level Highlights</h3>
                  <div className="token-stream">
                    {(result.text_highlights || []).map((item, index) => (
                      <span key={`${item.word}-${index}`} className={`token ${item.direction || 'neutral'}`}>
                        {item.word}
                      </span>
                    ))}
                  </div>

                  <ul className="insight-list">
                    {(result.explanation || []).slice(0, 10).map((item, index) => (
                      <li key={`${item.word}-${index}`}>
                        <strong>{item.word}</strong>
                        <span className={item.direction === 'positive' ? 'status-positive' : 'status-negative'}>
                          {item.direction}
                        </span>
                        <em>{item.weight.toFixed(4)}</em>
                      </li>
                    ))}
                  </ul>
                </article>
              )}

              {result.mode === 'aspect' && (
                <article className="result-card">
                  <h3>Aspect Breakdown</h3>
                  {result.aspects?.length ? (
                    <div className="aspect-grid">
                      {result.aspects.map((aspect) => (
                        <div key={aspect.aspect} className="aspect-card">
                          <strong>{aspect.aspect}</strong>
                          <span className={aspect.sentiment === 'positive' ? 'status-positive' : 'status-negative'}>
                            {aspect.sentiment}
                          </span>
                          <em>{formatPercent(aspect.confidence)}</em>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="helper-text">No explicit movie aspects detected in this review.</p>
                  )}

                  <div className="sentence-list">
                    {(result.sentences || []).map((entry, index) => (
                      <div key={`sentence-${index}`}>
                        <p>{entry.sentence}</p>
                        <small>{entry.sentiment} • {formatPercent(entry.confidence)}</small>
                      </div>
                    ))}
                  </div>
                </article>
              )}

              {result.mode === 'compare' && (
                <article className="result-card">
                  <div className="compare-heading">
                    <h3>Model Comparison</h3>
                    <span className="status-positive">Winner: {MODEL_LABELS[result.winner] || result.winner}</span>
                  </div>
                  <div className="compare-grid">
                    {Object.entries(result.models || {}).map(([modelName, modelResult]) => (
                      <div
                        key={modelName}
                        className={`compare-card ${result.winner === modelName ? 'winner' : ''}`}
                      >
                        <div className="compare-card-head">
                          <strong>{MODEL_LABELS[modelName] || modelName}</strong>
                          {!modelResult.available && <span className="tag-muted">Fallback</span>}
                        </div>
                        <span className={`sentiment-pill ${normalizeSentiment(modelResult.sentiment)}`}>{modelResult.sentiment}</span>
                        <div className="progress-track">
                          <div
                            className={`progress-fill ${normalizeSentiment(modelResult.sentiment)}`}
                            style={{ width: `${Math.max((modelResult.confidence || 0) * 100, 3)}%` }}
                          />
                        </div>
                        <div className="compare-meta">
                          <span>Confidence {formatPercent(modelResult.confidence)}</span>
                          <span>Pos {formatPercent(modelResult.positive_prob)}</span>
                          <span>Neg {formatPercent(modelResult.negative_prob)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </article>
              )}

              <article className="result-card">
                <div className="compare-heading">
                  <h3>Similar Review Search</h3>
                  <button
                    type="button"
                    className="btn btn-subtle"
                    onClick={handleFindSimilar}
                    disabled={similarLoading}
                  >
                    <FiSearch />
                    {similarLoading ? 'Searching...' : 'Find Similar in History'}
                  </button>
                </div>

                {similarError && <div className="status-banner error">{similarError}</div>}

                {!similarError && !similarLoading && similarReviews.length === 0 && (
                  <p className="helper-text">Run similarity search after analyzing to compare against saved history.</p>
                )}

                {similarReviews.length > 0 && (
                  <div className="similar-list">
                    {similarReviews.map((item) => (
                      <div key={item.id} className="similar-item">
                        <p>{item.review_text}</p>
                        <small>
                          Similarity {(item.similarity * 100).toFixed(1)}% • {item.sentiment} • {formatPercent(item.confidence)}
                        </small>
                      </div>
                    ))}
                  </div>
                )}
              </article>
            </div>
          )}
        </section>

        <section className="panel panel-batch">
          <div className="panel-heading">
            <h2>Batch Analyzer</h2>
            <p>Upload CSV or paste one review per line. Batch size: 5 to 50 reviews.</p>
          </div>

          <textarea
            className="review-input batch-input"
            value={batchText}
            onChange={(event) => setBatchText(event.target.value)}
            placeholder="Paste reviews, one per line"
          />

          <div className="batch-toolbar">
            <label className="btn btn-subtle batch-file-input">
              <FiUpload />
              Import CSV
              <input type="file" accept=".csv,.txt" onChange={handleBatchFileUpload} />
            </label>
            <button type="button" className="btn btn-primary" onClick={handleBatchAnalyze} disabled={batchLoading}>
              <FiActivity />
              {batchLoading ? 'Analyzing Batch...' : 'Run Batch Analysis'}
            </button>
            <button
              type="button"
              className="btn btn-subtle"
              onClick={handleBatchExport}
              disabled={!batchResult?.results?.length}
            >
              <FiDownload />
              Export CSV
            </button>
          </div>

          <div className="input-meta">
            <span>{batchReviews.length} reviews detected</span>
            <span>Limit: 50</span>
          </div>

          {batchProgress > 0 && (
            <div className="batch-progress-wrap">
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${batchProgress}%` }} />
              </div>
              <small>{batchProgress}% complete</small>
            </div>
          )}

          {batchError && <div className="status-banner error">{batchError}</div>}

          {batchResult && (
            <div className="batch-result">
              <div className="probability-grid">
                <div className="probability-card positive">
                  <span>Positive</span>
                  <strong>{batchResult.summary?.positive_count || 0}</strong>
                </div>
                <div className="probability-card negative">
                  <span>Negative</span>
                  <strong>{batchResult.summary?.negative_count || 0}</strong>
                </div>
              </div>
              <p className="helper-text">
                Average confidence: {formatPercent(batchResult.summary?.avg_confidence)}
              </p>
              <div className="table-wrap">
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Review</th>
                      <th>Sentiment</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(batchResult.results || []).map((item) => (
                      <tr key={`${item.index}-${item.review.slice(0, 12)}`}>
                        <td>{item.index + 1}</td>
                        <td>{item.review}</td>
                        <td>
                          <span className={statusClassName(item.sentiment)}>
                            {item.sentiment}
                          </span>
                        </td>
                        <td>{formatPercent(item.confidence)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
      </div>
    </motion.div>
  );
}

export default AnalyzerPage;
