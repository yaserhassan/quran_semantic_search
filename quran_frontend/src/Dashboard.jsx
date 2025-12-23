import { useState } from "react";
import { searchVerses } from "./api";

function Dashboard({ token, onLogout }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [total, setTotal] = useState(0);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const RESULTS_PER_PAGE = 10;
  const [currentPage, setCurrentPage] = useState(1);

  const totalPages = Math.max(1, Math.ceil(total / RESULTS_PER_PAGE));

  // ===== SEARCH =====
  const handleSearch = async (e) => {
    e.preventDefault();

    const trimmed = query.trim();
    if (!trimmed) {
      setError("Please enter a word or concept.");
      return;
    }
    if (!token) {
      setError("You are not logged in.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      setCurrentPage(1);
      const data = await searchVerses(
        token,
        trimmed,
        1,
        RESULTS_PER_PAGE
      );
      setResults(data.results); // نتائج الصفحة فقط
      setTotal(data.total);     // العدد الكلي من السيرفر
    } catch (err) {
      setError(err.message || "Search failed");
    } finally {
      setLoading(false);
    }
  };

  // ===== PAGINATION =====
  const goToPage = async (p) => {
    const trimmed = query.trim();
    if (!trimmed) return;

    const page = Math.min(Math.max(1, p), totalPages);

    setLoading(true);
    setError("");
    try {
      const data = await searchVerses(
        token,
        trimmed,
        page,
        RESULTS_PER_PAGE
      );
      setResults(data.results);
      setTotal(data.total);
      setCurrentPage(page);
    } catch (err) {
      setError(err.message || "Search failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-bg">
      <div className="dashboard-shell">
        {/* Header */}
        <div className="dashboard-header">
          <h1 className="dashboard-title">Qur&apos;an Semantic Search</h1>
          <button className="btn-outline" onClick={onLogout}>
            Logout
          </button>
        </div>

        {/* Intro */}
        <div className="dashboard-intro">
          <p>
            This system provides <b>semantic search</b> for Qur&apos;anic verses.
          </p>
          <p>
            You can type a word in <b>Arabic</b> or <b>English</b>, and the system
            will retrieve the most related verses by <b>meaning</b>.
          </p>
        </div>

        {/* Search */}
        <form className="dashboard-search-form" onSubmit={handleSearch}>
          <input
            className="form-input"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your question..."
          />
          <button
            type="submit"
            disabled={loading}
            className="btn-primary"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </form>

        {/* Error */}
        {error && <div className="alert-error">{error}</div>}

        {/* Results */}
        <div className="dashboard-results">
          {results.length === 0 && !loading && !error && (
            <p className="no-results-text">
              No results yet. Try searching for a concept like <b>hearing</b> or{" "}
              <b>بصر</b>.
            </p>
          )}

          {total > 0 && (
            <p className="results-summary">
              Found <b>{total}</b> verses for query:{" "}
              <span className="results-query">{query}</span>
            </p>
          )}

          <div className="results-list">
            {results.map((r) => (
              <div
                key={`${r.ref}-${r.rank}`}
                className="result-card"
              >
                <div className="result-meta">
                  <span>
                    Rank {r.rank} • Ref {r.ref}
                  </span>
                </div>

                <p className="arabic-verse">{r.arabic}</p>
                <p className="english-verse">{r.english}</p>
              </div>
            ))}
          </div>

          {total > 0 && (
            <div className="pagination-row">
              <span>
                Page {currentPage} of {totalPages}
              </span>

              {totalPages > 1 && (
                <div className="pagination-buttons">
                  <button
                    onClick={() => goToPage(currentPage - 1)}
                    disabled={currentPage === 1 || loading}
                    className="pagination-btn"
                  >
                    Previous
                  </button>

                  <button
                    onClick={() => goToPage(currentPage + 1)}
                    disabled={currentPage === totalPages || loading}
                    className="pagination-btn"
                  >
                    Next
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
