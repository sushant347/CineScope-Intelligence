import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  FiFilter,
  FiRefreshCw,
  FiShare2,
  FiThumbsDown,
  FiThumbsUp,
  FiTrendingDown,
  FiTrendingUp,
} from 'react-icons/fi';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  createPredictionShare,
  getPredictionStats,
  getPredictionTokens,
  getPredictions,
  updatePredictionFeedback,
} from '../services/api';

const formatPercent = (value) => `${((value || 0) * 100).toFixed(1)}%`;

const defaultFilters = {
  page: 1,
  page_size: 12,
  sentiment: '',
  model: '',
  feedback: '',
  ordering: '-created_at',
  q: '',
  start_date: '',
  end_date: '',
};

function HistoryPage() {
  const [predictions, setPredictions] = useState([]);
  const [stats, setStats] = useState(null);
  const [tokens, setTokens] = useState([]);
  const [tokenSentiment, setTokenSentiment] = useState('all');
  const [showTopKeywords, setShowTopKeywords] = useState(false);
  const [filters, setFilters] = useState(defaultFilters);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [actionMessage, setActionMessage] = useState('');

  const loadData = async () => {
    setLoading(true);
    setError('');

    try {
      const tokenParams = {
        limit: 32,
        sentiment: tokenSentiment === 'all' ? '' : tokenSentiment,
      };

      const [predictionResponse, statsResponse, tokenResponse] = await Promise.all([
        getPredictions(filters),
        getPredictionStats(),
        getPredictionTokens(tokenParams),
      ]);

      const predictionPayload = predictionResponse.data;
      const rows = predictionPayload.results || predictionPayload || [];

      setPredictions(rows);
      setTotalCount(predictionPayload.count || rows.length || 0);
      setStats(statsResponse.data);
      setTokens(tokenResponse.data.tokens || []);
    } catch (err) {
      setError(err.response?.data?.detail || 'Unable to load history. Please log in again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [filters, tokenSentiment]);

  useEffect(() => {
    if (!actionMessage) {
      return undefined;
    }

    const timeout = window.setTimeout(() => setActionMessage(''), 2600);
    return () => window.clearTimeout(timeout);
  }, [actionMessage]);

  const distribution = useMemo(() => {
    if (!stats || !stats.total_predictions) {
      return { positive: 0, negative: 0 };
    }

    return {
      positive: (stats.positive_count / stats.total_predictions) * 100,
      negative: (stats.negative_count / stats.total_predictions) * 100,
    };
  }, [stats]);

  const confidenceHistogram = useMemo(() => {
    const bins = [
      { label: '0.0-0.2', count: 0 },
      { label: '0.2-0.4', count: 0 },
      { label: '0.4-0.6', count: 0 },
      { label: '0.6-0.8', count: 0 },
      { label: '0.8-1.0', count: 0 },
    ];

    predictions.forEach((item) => {
      const confidence = Number(item.confidence || 0);
      const index = Math.min(Math.floor(confidence * 5), 4);
      bins[index].count += 1;
    });

    return bins;
  }, [predictions]);

  const aspectData = useMemo(() => {
    const counts = {};
    predictions.forEach((prediction) => {
      (prediction.aspects || []).forEach((aspectItem) => {
        const aspectName = aspectItem.aspect || 'unknown';
        counts[aspectName] = (counts[aspectName] || 0) + 1;
      });
    });

    return Object.entries(counts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 8);
  }, [predictions]);

  const totalPages = useMemo(() => {
    const pageSize = Number(filters.page_size || 1);
    return Math.max(1, Math.ceil(totalCount / pageSize));
  }, [totalCount, filters.page_size]);

  const maxTokenCount = useMemo(
    () => tokens.reduce((max, token) => Math.max(max, token.count || 0), 1),
    [tokens]
  );

  const updateFilter = (name, value) => {
    setFilters((previous) => ({
      ...previous,
      page: 1,
      [name]: value,
    }));
  };

  const goToPage = (page) => {
    setFilters((previous) => ({
      ...previous,
      page,
    }));
  };

  const handleFeedback = async (predictionId, userCorrect) => {
    try {
      await updatePredictionFeedback(predictionId, { user_correct: userCorrect });
      setPredictions((previous) =>
        previous.map((prediction) => (
          prediction.id === predictionId
            ? { ...prediction, user_correct: userCorrect }
            : prediction
        ))
      );
      setActionMessage('Feedback saved.');
    } catch {
      setActionMessage('Could not save feedback right now.');
    }
  };

  const handleShare = async (predictionId) => {
    try {
      const response = await createPredictionShare(predictionId);
      const sharePath = response.data.frontend_path || `/shared/${response.data.share_uuid}`;
      const shareUrl = `${window.location.origin}${sharePath}`;

      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(shareUrl);
        setActionMessage('Share link copied to clipboard.');
      } else {
        setActionMessage(`Share link: ${shareUrl}`);
      }
    } catch {
      setActionMessage('Could not create a share link.');
    }
  };

  return (
    <motion.div
      className="history-page"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
    >
      <div className="container history-shell">
        <header className="history-header">
          <div>
            <h1>Prediction History</h1>
            <p>Track the sentiment profile of your analyzed reviews.</p>
          </div>
          <button type="button" className="btn btn-subtle" onClick={() => loadData()}>
            <FiRefreshCw />
            Refresh
          </button>
        </header>

        <section className="history-filters panel">
          <div className="history-filter-title">
            <h2>
              <FiFilter />
              Filter History
            </h2>
            <button
              type="button"
              className="btn btn-subtle"
              onClick={() => setFilters(defaultFilters)}
            >
              Reset
            </button>
          </div>

          <div className="filters-grid">
            <input
              value={filters.q}
              onChange={(event) => updateFilter('q', event.target.value)}
              placeholder="Search review text"
            />
            <select
              value={filters.sentiment}
              onChange={(event) => updateFilter('sentiment', event.target.value)}
            >
              <option value="">All sentiments</option>
              <option value="positive">Positive</option>
              <option value="negative">Negative</option>
            </select>
            <select
              value={filters.model}
              onChange={(event) => updateFilter('model', event.target.value)}
            >
              <option value="">All models</option>
              {(stats?.model_usage || []).map((item) => (
                <option key={item.model_used} value={item.model_used}>{item.model_used}</option>
              ))}
            </select>
            <select
              value={filters.feedback}
              onChange={(event) => updateFilter('feedback', event.target.value)}
            >
              <option value="">All feedback states</option>
              <option value="corrected">Corrected</option>
              <option value="uncorrected">Not corrected</option>
            </select>
            <input
              type="date"
              value={filters.start_date}
              onChange={(event) => updateFilter('start_date', event.target.value)}
            />
            <input
              type="date"
              value={filters.end_date}
              onChange={(event) => updateFilter('end_date', event.target.value)}
            />
            <select
              value={filters.ordering}
              onChange={(event) => updateFilter('ordering', event.target.value)}
            >
              <option value="-created_at">Newest first</option>
              <option value="created_at">Oldest first</option>
              <option value="-confidence">Highest confidence</option>
              <option value="confidence">Lowest confidence</option>
            </select>
          </div>
        </section>

        {loading && (
          <div className="result-empty">
            <div className="spinner"></div>
            <p>Loading your analytics...</p>
          </div>
        )}

        {!loading && error && <div className="status-banner error">{error}</div>}

        {!loading && !error && stats && (
          <>
            {actionMessage && <div className="status-banner">{actionMessage}</div>}

            <section className="insight-grid">
              <article className="insight-card">
                <span>Total Predictions</span>
                <strong>{stats.total_predictions}</strong>
              </article>
              <article className="insight-card">
                <span>Average Confidence</span>
                <strong>{formatPercent(stats.avg_confidence)}</strong>
              </article>
              <article className="insight-card positive">
                <span>Positive Reviews</span>
                <strong>{stats.positive_count}</strong>
                <small><FiTrendingUp /> {stats.positive_percentage}%</small>
              </article>
              <article className="insight-card negative">
                <span>Negative Reviews</span>
                <strong>{stats.negative_count}</strong>
                <small><FiTrendingDown /> {stats.negative_percentage}%</small>
              </article>
            </section>

            <section className="distribution-card">
              <h2>Sentiment Distribution</h2>
              <div className="distribution-track">
                <div className="distribution-positive" style={{ width: `${distribution.positive}%` }} />
                <div className="distribution-negative" style={{ width: `${distribution.negative}%` }} />
              </div>
              <div className="distribution-legend">
                <span className="status-positive">Positive {distribution.positive.toFixed(1)}%</span>
                <span className="status-negative">Negative {distribution.negative.toFixed(1)}%</span>
              </div>
            </section>

            <section className="chart-grid">
              <article className="history-table-card">
                <h2>Sentiment Timeline (30 Days)</h2>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={stats.trend || []}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="day" />
                      <YAxis allowDecimals={false} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="positive" stroke="#1f8d56" strokeWidth={2} />
                      <Line type="monotone" dataKey="negative" stroke="#c54f40" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </article>

              <article className="history-table-card">
                <h2>Aspect Mentions</h2>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={aspectData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis allowDecimals={false} />
                      <Tooltip />
                      <Bar dataKey="value" fill="#0f766e" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </article>

              <article className="history-table-card">
                <h2>Confidence Histogram</h2>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={confidenceHistogram}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="label" />
                      <YAxis allowDecimals={false} />
                      <Tooltip />
                      <Bar dataKey="count" fill="#b35e16" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </article>
            </section>

            <section className="history-table-card">
              <div className="token-heading">
                <h2>Top Keywords</h2>
                <div className="token-controls">
                  {showTopKeywords && (
                    <select
                      value={tokenSentiment}
                      onChange={(event) => setTokenSentiment(event.target.value)}
                    >
                      <option value="all">All</option>
                      <option value="positive">Positive only</option>
                      <option value="negative">Negative only</option>
                    </select>
                  )}
                  <button
                    type="button"
                    className="btn btn-subtle"
                    onClick={() => setShowTopKeywords((previous) => !previous)}
                  >
                    {showTopKeywords ? 'Hide' : 'Show'}
                  </button>
                </div>
              </div>

              {showTopKeywords && (
                <>
                  {tokens.length > 0 ? (
                    <ul className="token-list">
                      {tokens.map((token) => {
                        const width = Math.max(Math.round((token.count / maxTokenCount) * 100), 8);
                        return (
                          <li key={token.word} className="token-list-item">
                            <span className="token-word">{token.word}</span>
                            <div className="token-bar-track">
                              <div className="token-bar-fill" style={{ width: `${width}%` }} />
                            </div>
                            <em>{token.count}</em>
                          </li>
                        );
                      })}
                    </ul>
                  ) : (
                    <p className="helper-text">No token data yet. Run more analyses first.</p>
                  )}
                </>
              )}
            </section>

            <section className="history-table-card">
              <h2>Recent Entries</h2>
              {predictions.length > 0 ? (
                <div className="table-wrap">
                  <table className="history-table">
                    <thead>
                      <tr>
                        <th>Review</th>
                        <th>Sentiment</th>
                        <th>Confidence</th>
                        <th>Model</th>
                        <th>Feedback</th>
                        <th>Share</th>
                        <th>Date</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictions.map((prediction) => (
                        <tr key={prediction.id}>
                          <td>{prediction.review_text}</td>
                          <td>
                            <span className={prediction.sentiment === 'positive' ? 'status-positive' : 'status-negative'}>
                              {prediction.sentiment}
                            </span>
                          </td>
                          <td>{formatPercent(prediction.confidence)}</td>
                          <td>{prediction.model_used}</td>
                          <td>
                            <div className="row-action-group">
                              <button
                                type="button"
                                className={`row-chip ${prediction.user_correct === 'positive' ? 'active' : ''}`}
                                onClick={() => handleFeedback(prediction.id, 'positive')}
                              >
                                <FiThumbsUp />
                              </button>
                              <button
                                type="button"
                                className={`row-chip ${prediction.user_correct === 'negative' ? 'active' : ''}`}
                                onClick={() => handleFeedback(prediction.id, 'negative')}
                              >
                                <FiThumbsDown />
                              </button>
                            </div>
                          </td>
                          <td>
                            <button
                              type="button"
                              className="row-chip"
                              onClick={() => handleShare(prediction.id)}
                            >
                              <FiShare2 />
                            </button>
                          </td>
                          <td>{new Date(prediction.created_at).toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="helper-text">No predictions saved yet. Run analyses in the Analyzer page first.</p>
              )}

              <div className="pagination-row">
                <button
                  type="button"
                  className="btn btn-subtle"
                  onClick={() => goToPage(Math.max(filters.page - 1, 1))}
                  disabled={filters.page <= 1}
                >
                  Previous
                </button>
                <span>Page {filters.page} of {totalPages}</span>
                <button
                  type="button"
                  className="btn btn-subtle"
                  onClick={() => goToPage(Math.min(filters.page + 1, totalPages))}
                  disabled={filters.page >= totalPages}
                >
                  Next
                </button>
              </div>
            </section>
          </>
        )}
      </div>
    </motion.div>
  );
}

export default HistoryPage;
