import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FiArrowLeft } from 'react-icons/fi';

function NotFoundPage() {
  return (
    <div className="auth-shell">
      <motion.section
        className="auth-card"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0 }}
      >
        <h1>Page not found</h1>
        <p>The link you opened is not available in this deployment. Return to the main workspace.</p>
        <Link to="/" className="btn btn-primary">
          <FiArrowLeft />
          Back to overview
        </Link>
      </motion.section>
    </div>
  );
}

export default NotFoundPage;
