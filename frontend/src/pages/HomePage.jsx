import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  FiActivity,
  FiArrowRight,
  FiCpu,
  FiDatabase,
  FiLayers,
  FiSearch,
  FiShield,
  FiTrendingUp,
} from 'react-icons/fi';

const capabilities = [
  {
    icon: <FiActivity />,
    title: 'Fast Inference API',
    description: 'Low-latency sentiment scoring with confidence probabilities and robust fallbacks.',
  },
  {
    icon: <FiSearch />,
    title: 'Explainable Outputs',
    description: 'Word-level attribution and interpretable highlights powered by LIME.',
  },
  {
    icon: <FiLayers />,
    title: 'Aspect Insights',
    description: 'Sentence-by-sentence aspect polarity for acting, story, direction, and more.',
  },
  {
    icon: <FiDatabase />,
    title: 'Dataset Driven',
    description: 'Built on the IMDB 50K corpus with a clean preprocessing and baseline pipeline.',
  },
  {
    icon: <FiCpu />,
    title: 'Multi-Model Inference',
    description: 'From classical TF-IDF baselines to transformer-based BERT fine-tuning.',
  },
  {
    icon: <FiShield />,
    title: 'Deployment Ready',
    description: 'Structured for Neon + Render deployment with production-safe backend settings.',
  },
];

const metrics = [
  { label: 'Reviews in Dataset', value: '50,000' },
  { label: 'Baseline Accuracy', value: '90.2%' },
  { label: 'BERT (CPU-safe run)', value: '86.0%' },
  { label: 'API Modes', value: '3' },
];

const workflow = [
  {
    title: 'Preprocess and Train',
    description: 'Tokenize, normalize, and train baseline plus transformer models with reproducible scripts.',
  },
  {
    title: 'Serve Through Django API',
    description: 'Expose prediction, explanation, and aspect endpoints with JWT-based auth and history tracking.',
  },
  {
    title: 'Visualize in React',
    description: 'Present confidence, explanation, and analytics in a clean operational interface.',
  },
];

const itemVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

function HomePage() {
  return (
    <motion.div
      className="home-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <section className="hero-section">
        <div className="container hero-grid">
          <motion.div
            className="hero-copy"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55 }}
          >
            <span className="eyebrow">Production NLP Application</span>
            <h1>Turn raw movie reviews into explainable sentiment intelligence.</h1>
            <p>
              CineScope Intelligence combines classical machine learning and transformer models
              to deliver professional sentiment analysis with transparent reasoning and clean analytics.
            </p>
            <div className="hero-actions">
              <Link to="/analyze" className="btn btn-primary">
                Open Analyzer <FiArrowRight />
              </Link>
              <Link to="/register" className="btn btn-subtle">Create Account</Link>
            </div>
          </motion.div>

          <motion.div
            className="hero-metric-grid"
            variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.08 } } }}
            initial="hidden"
            animate="visible"
          >
            {metrics.map((metric) => (
              <motion.div key={metric.label} className="metric-card" variants={itemVariants}>
                <span>{metric.label}</span>
                <strong>{metric.value}</strong>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      <section className="section-block">
        <div className="container">
          <div className="section-heading">
            <h2>Core capabilities</h2>
            <p>Built for reliable inference, transparent outputs, and practical deployment.</p>
          </div>

          <motion.div
            className="feature-grid"
            variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.1 } } }}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.2 }}
          >
            {capabilities.map((feature) => (
              <motion.article key={feature.title} className="feature-card" variants={itemVariants}>
                <div className="feature-icon">{feature.icon}</div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
              </motion.article>
            ))}
          </motion.div>
        </div>
      </section>

      <section className="section-block section-tinted">
        <div className="container">
          <div className="section-heading">
            <h2>How it works</h2>
            <p>An end-to-end workflow from training to API serving and interactive analytics.</p>
          </div>

          <div className="workflow-grid">
            {workflow.map((step, index) => (
              <motion.article
                key={step.title}
                className="workflow-card"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.08 }}
              >
                <span>{`0${index + 1}`}</span>
                <h3>{step.title}</h3>
                <p>{step.description}</p>
              </motion.article>
            ))}
          </div>
        </div>
      </section>

      <section className="section-block">
        <div className="container">
          <motion.div
            className="cta-card"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2>Ready to run live inference?</h2>
            <p>Launch the analyzer, test your own reviews, and monitor behavior across models.</p>
            <div className="cta-actions">
              <Link to="/analyze" className="btn btn-primary">
                Start Analysis <FiArrowRight />
              </Link>
              <Link to="/history" className="btn btn-subtle">
                <FiTrendingUp /> View Analytics
              </Link>
            </div>
            <Link to="/register" className="cta-link">
              Register to save and track predictions
            </Link>
          </motion.div>
        </div>
      </section>
    </motion.div>
  );
}

export default HomePage;
