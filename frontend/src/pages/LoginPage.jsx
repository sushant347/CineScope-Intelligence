import { useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { FiLogIn } from 'react-icons/fi';
import { login } from '../services/api';

function LoginPage({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!username || !password) {
      setError('Username and password are required.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await login(username, password);
      onLogin(username, response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Authentication failed. Verify your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-shell">
      <motion.section
        className="auth-card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0 }}
      >
        <h1>Welcome back</h1>
        <p>Sign in to access saved predictions and historical analytics.</p>

        {error && <div className="status-banner error">{error}</div>}

        <form onSubmit={handleSubmit} className="auth-form">
          <label htmlFor="username">Username</label>
          <input
            id="username"
            type="text"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            placeholder="Enter your username"
            autoComplete="username"
          />

          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder="Enter your password"
            autoComplete="current-password"
          />

          <button type="submit" className="btn btn-primary" disabled={loading}>
            <FiLogIn />
            {loading ? 'Signing In...' : 'Sign In'}
          </button>
        </form>

        <div className="auth-footnote">
          New here? <Link to="/register">Create an account</Link>
        </div>
      </motion.section>
    </div>
  );
}

export default LoginPage;
