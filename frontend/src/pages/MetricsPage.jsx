import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
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
import { FiActivity, FiDatabase, FiTarget } from 'react-icons/fi';
import { getPredictionMetrics } from '../services/api';

const asPercent = (value) => `${((value || 0) * 100).toFixed(1)}%`;

function MetricsPage() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const loadMetrics = async () => {
      setLoading(true);
      setError('');

      try {
        const response = await getPredictionMetrics();
        setMetrics(response.data);
      } catch (err) {
        setError(err.response?.data?.detail || 'Unable to load model metrics.');
      } finally {
        setLoading(false);
      }
    };

    loadMetrics();
  }, []);

  const usageChart = useMemo(() => {
    return (metrics?.model_usage || []).map((item) => ({
      model: item.model_used,
      count: item.count,
    }));
  }, [metrics]);

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
            <h1>Model Metrics</h1>
            <p>Calibration, usage, and reliability diagnostics for your inference history.</p>
          </div>
        </header>

        {loading && (
          <div className="result-empty">
            <div className="spinner"></div>
            <p>Loading metrics...</p>
          </div>
        )}

        {!loading && error && <div className="status-banner error">{error}</div>}

        {!loading && !error && metrics && (
          <>
            <section className="insight-grid">
              <article className="insight-card">
                <span>Total Predictions</span>
                <strong>{metrics.total_predictions}</strong>
                <small><FiDatabase /> User history coverage</small>
              </article>
              <article className="insight-card">
                <span>Feedback Coverage</span>
                <strong>{asPercent(metrics.feedback_coverage)}</strong>
                <small><FiActivity /> Corrections available</small>
              </article>
              <article className="insight-card">
                <span>Brier Score</span>
                <strong>{Number(metrics.brier_score || 0).toFixed(4)}</strong>
                <small><FiTarget /> Lower is better</small>
              </article>
            </section>

            <section className="chart-grid">
              <article className="history-table-card">
                <h2>Calibration Curve</h2>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={metrics.calibration || []}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="bin" />
                      <YAxis domain={[0, 1]} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="predicted" stroke="#0f766e" strokeWidth={2} />
                      <Line type="monotone" dataKey="observed" stroke="#b35e16" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </article>

              <article className="history-table-card">
                <h2>Model Usage</h2>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={usageChart}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="model" />
                      <YAxis allowDecimals={false} />
                      <Tooltip />
                      <Bar dataKey="count" fill="#0f766e" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </article>
            </section>

            <section className="history-table-card">
              <h2>Model Card</h2>
              <div className="model-grid">
                {(metrics.model_catalog || []).map((item) => (
                  <article key={item.id} className="model-card">
                    <h3>{item.name}</h3>
                    <p>{item.family}</p>
                    <span className="status-positive">{item.status}</span>
                    <small>Reported accuracy: {(item.reported_accuracy * 100).toFixed(1)}%</small>
                  </article>
                ))}
              </div>
            </section>
          </>
        )}
      </div>
    </motion.div>
  );
}

export default MetricsPage;
