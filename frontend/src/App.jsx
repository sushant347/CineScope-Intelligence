import { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink, Navigate } from 'react-router-dom';
import { FiFilm, FiLogIn, FiLogOut, FiUserPlus } from 'react-icons/fi';
import HomePage from './pages/HomePage';
import AnalyzerPage from './pages/AnalyzerPage';
import HistoryPage from './pages/HistoryPage';
import MetricsPage from './pages/MetricsPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import NotFoundPage from './pages/NotFoundPage';
import SharedPredictionPage from './pages/SharedPredictionPage';

const navClass = ({ isActive }) => `nav-link ${isActive ? 'active' : ''}`;

function App() {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    const username = localStorage.getItem('username');
    if (token && username) {
      setUser({ username });
    }
    setIsLoading(false);
  }, []);

  const handleLogin = (userData, tokens) => {
    localStorage.setItem('access_token', tokens.access);
    localStorage.setItem('refresh_token', tokens.refresh);
    localStorage.setItem('username', userData.username || userData);
    setUser({ username: userData.username || userData });
  };

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('username');
    setUser(null);
  };

  if (isLoading) {
    return (
      <div className="loading-spinner page-center">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <Router>
      <div className="site-shell">
        <header className="topbar">
          <div className="container topbar-inner">
            <NavLink to="/" className="brand">
              <span className="brand-mark">
                <FiFilm />
              </span>
              <span className="brand-copy">
                <strong>CineScope Intelligence</strong>
                <small>Film Review Sentiment Platform</small>
              </span>
            </NavLink>

            <nav className="primary-nav">
              <NavLink to="/" end className={navClass}>Overview</NavLink>
              <NavLink to="/analyze" className={navClass}>Analyzer</NavLink>
              {user && <NavLink to="/history" className={navClass}>History</NavLink>}
              {user && <NavLink to="/metrics" className={navClass}>Metrics</NavLink>}
            </nav>

            <div className="auth-actions">
              {user ? (
                <>
                  <span className="user-chip">{user.username}</span>
                  <button className="btn btn-subtle" onClick={handleLogout}>
                    <FiLogOut />
                    Logout
                  </button>
                </>
              ) : (
                <>
                  <NavLink to="/login" className="btn btn-subtle">
                    <FiLogIn />
                    Login
                  </NavLink>
                  <NavLink to="/register" className="btn btn-primary">
                    <FiUserPlus />
                    Create Account
                  </NavLink>
                </>
              )}
            </div>
          </div>
        </header>

        <main className="page-main">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/analyze" element={<AnalyzerPage />} />
            <Route
              path="/history"
              element={user ? <HistoryPage /> : <Navigate to="/login" />}
            />
            <Route
              path="/metrics"
              element={user ? <MetricsPage /> : <Navigate to="/login" />}
            />
            <Route
              path="/login"
              element={user ? <Navigate to="/analyze" /> : <LoginPage onLogin={handleLogin} />}
            />
            <Route
              path="/register"
              element={user ? <Navigate to="/analyze" /> : <RegisterPage onLogin={handleLogin} />}
            />
            <Route path="/shared/:shareId" element={<SharedPredictionPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </main>

        <footer className="site-footer">
          <div className="container site-footer-inner">
            <p>Movie Review Sentiment System • Django, React, NLP and Transformers</p>
            <p>Ready for Neon + Render deployment</p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
