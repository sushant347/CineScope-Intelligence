import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FiExternalLink } from 'react-icons/fi';
import { getSharedPrediction } from '../services/api';

const formatPercent = (value) => `${((value || 0) * 100).toFixed(1)}%`;

function SharedPredictionPage() {
  const { shareId } = useParams();
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const loadSharedPrediction = async () => {
      setLoading(true);
      setError('');

      try {
        const response = await getSharedPrediction(shareId);
        setPrediction(response.data);
      } catch {
        setError('Shared prediction link is invalid or no longer public.');
      } finally {
        setLoading(false);
      }
    };

    loadSharedPrediction();
  }, [shareId]);

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
            <h1>Shared Prediction</h1>
            <p>Public snapshot of a sentiment analysis result.</p>
          </div>
        </header>

        {loading && (
          <div className="result-empty">
            <div className="spinner"></div>
            <p>Loading shared prediction...</p>
          </div>
        )}

        {!loading && error && <div className="status-banner error">{error}</div>}

        {!loading && !error && prediction && (
          <section className="history-table-card">
            <div className="result-header">
              <div>
                <h2>Prediction Summary</h2>
                <p>Model: {prediction.model_used}</p>
              </div>
              <span className={`sentiment-pill ${prediction.sentiment}`}>{prediction.sentiment}</span>
            </div>

            <p className="shared-review">{prediction.review_text}</p>

            <div className="probability-grid">
              <div className="probability-card positive">
                <span>Positive Probability</span>
                <strong>{formatPercent(prediction.positive_prob)}</strong>
              </div>
              <div className="probability-card negative">
                <span>Negative Probability</span>
                <strong>{formatPercent(prediction.negative_prob)}</strong>
              </div>
            </div>

            <p className="helper-text">Confidence: {formatPercent(prediction.confidence)}</p>
            <Link to="/analyze" className="btn btn-primary shared-cta">
              <FiExternalLink />
              Run Your Own Analysis
            </Link>
          </section>
        )}
      </div>
    </motion.div>
  );
}

export default SharedPredictionPage;
