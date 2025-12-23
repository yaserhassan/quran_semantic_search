import { useEffect, useMemo, useState } from "react";
import { searchVerses } from "./api";

function Dashboard({ token, onLogout }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const RESULTS_PER_PAGE = 10;
  const [currentPage, setCurrentPage] = useState(1);

  const totalPages = useMemo(() => {
    return Math.max(1, Math.ceil(results.length / RESULTS_PER_PAGE));
  }, [results.length]);

  // لو النتائج تغيرت وصارت الصفحات أقل، نزّل currentPage تلقائياً
  useEffect(() => {
    if (currentPage > totalPages) setCurrentPage(totalPages);
  }, [results.length, totalPages, currentPage]);

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
      const data = await searchVerses(token, trimmed);
      const arr = Array.isArray(data.results) ? data.results : [];
      setResults(arr);
      setCurrentPage(1);
    } catch (err) {
      setResults([]);
      setCurrentPage(1);
      setError(err.message || "Search failed");
    } finally {
      setLoading(false);
    }
  };

  const goToPage = (p) => {
    const page = Math.min(Math.max(1, p), totalPages);
    setCurrentPage(page);
  };

  const start = (currentPage - 1) * RESULTS_PER_PAGE;
  const end = start + RESULTS_PER_PAGE;
  const pageResults = results.slice(start, end);

  return (
    <div className="page-bg">
      <div className="dashboard-shell">
        <div className="dashboard-header">
          <h1 className="dashboard-title">Qur&apos;an Semantic Search</h1>
          <button type="button" className="btn-outline" onClick={onLogout}>
            Logout
          </button>
        </div>

        <div className="dashboard-intro">
          <p>
            This system provides <b>semantic search</b> for Qur&apos;anic verses.
          </p>
          <p>
            You can type a word in <b>Arabic</b> or <b>English</b>, and the system
            will retrieve the most related verses by <b>meaning</b>.
          </p>
        </div>

        <form className="dashboard-search-form" onSubmit={handleSearch}>
          <input
            className="form-input"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your question..."
          />
          <button type="submit" disabled={loading} className="btn-primary">
            {loading ? "Searching..." : "Search"}
          </button>
        </form>

        {error && <div className="alert-error">{error}</div>}

        <div className="dashboard-results">
          {results.length === 0 && !loading && !error && (
            <p className="no-results-text">
              No results yet. Try searching for a concept like <b>hearing</b> or{" "}
              <b>بصر</b>.
            </p>
          )}

          {results.length > 0 && (
            <p className="results-summary">
              Found <b>{results.length}</b> verses for query:{" "}
              <span className="results-query">{query}</span>
            </p>
          )}

          <div className="results-list">
            {pageResults.map((r) => (
              <div key={`${r.ref}-${r.rank}`} className="result-card">
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

          {results.length > 0 && (
            <div className="pagination-row">
              <span>
                Page {currentPage} of {totalPages}
              </span>

              {totalPages > 1 && (
                <div className="pagination-buttons">
                  <button
                    type="button"
                    onClick={() => goToPage(currentPage - 1)}
                    disabled={currentPage === 1 || loading}
                    className="pagination-btn"
                  >
                    Previous
                  </button>

                  <button
                    type="button"
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
