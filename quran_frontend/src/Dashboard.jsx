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
      const data = await searchVerses(token, trimmed);
      setResults(data.results); // كل النتائج مرة واحدة
      setTotal(data.total);     // العدد النهائي
      setCurrentPage(1);        // نرجع للصفحة الأولى
    } catch (err) {
      setError(err.message || "Search failed");
    } finally {
      setLoading(false);
    }
  };

  // ===== PAGINATION (محلي) =====
  const goToPage = (p) => {
    const page = Math.min(Math.max(1, p), totalPages);
    setCurrentPage(page);
  };

  // ===== عرض نتائج الصفحة الحالية =====
  const start = (currentPage - 1) * RESULTS_PER_PAGE;
  const end = start + RESULTS_PER_PAGE;
  const pageResults = results.slice(start, end);

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

          {total > 0 && (
            <div className="pagination-row">
              <span>
                Page {currentPage} of {totalPages}
              </span>

              {totalPages > 1 && (
                <div className="pagination-buttons">
                  <button
                    onClick={() => goToPage(currentPage - 1)}
                    disabled={currentPage === 1}
                    className="pagination-btn"
                  >
                    Previous
                  </button>

                  <button
                    onClick={() => goToPage(currentPage + 1)}
                    disabled={currentPage === totalPages}
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
