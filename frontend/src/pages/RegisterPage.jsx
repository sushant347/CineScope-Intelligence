import { useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { FiUserPlus } from 'react-icons/fi';
import { login, register } from '../services/api';

function RegisterPage({ onLogin }) {
  const [form, setForm] = useState({
    username: '',
    email: '',
    password: '',
    password2: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleChange = (event) => {
    setForm((previous) => ({
      ...previous,
      [event.target.name]: event.target.value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    const { username, email, password, password2 } = form;

    if (!username || !email || !password || !password2) {
      setError('All fields are required.');
      return;
    }

    if (password !== password2) {
      setError('Passwords do not match.');
      return;
    }

    if (password.length < 6) {
      setError('Use at least 6 characters for the password.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      await register(username, email, password, password2);
      const loginResponse = await login(username, password);
      onLogin(username, loginResponse.data);
    } catch (err) {
      const data = err.response?.data;
      if (data && typeof data === 'object') {
        const messages = Object.values(data).flat().join(' ');
        setError(messages || 'Registration failed.');
      } else {
        setError('Registration failed.');
      }
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
        <h1>Create your account</h1>
        <p>Save analyses, review trends over time, and build your own sentiment log.</p>

        {error && <div className="status-banner error">{error}</div>}

        <form onSubmit={handleSubmit} className="auth-form">
          <label htmlFor="username">Username</label>
          <input
            id="username"
            name="username"
            type="text"
            value={form.username}
            onChange={handleChange}
            placeholder="Choose a username"
          />

          <label htmlFor="email">Email</label>
          <input
            id="email"
            name="email"
            type="email"
            value={form.email}
            onChange={handleChange}
            placeholder="you@example.com"
          />

          <label htmlFor="password">Password</label>
          <input
            id="password"
            name="password"
            type="password"
            value={form.password}
            onChange={handleChange}
            placeholder="At least 6 characters"
          />

          <label htmlFor="password2">Confirm Password</label>
          <input
            id="password2"
            name="password2"
            type="password"
            value={form.password2}
            onChange={handleChange}
            placeholder="Repeat password"
          />

          <button type="submit" className="btn btn-primary" disabled={loading}>
            <FiUserPlus />
            {loading ? 'Creating Account...' : 'Create Account'}
          </button>
        </form>

        <div className="auth-footnote">
          Already registered? <Link to="/login">Sign in</Link>
        </div>
      </motion.section>
    </div>
  );
}

export default RegisterPage;
