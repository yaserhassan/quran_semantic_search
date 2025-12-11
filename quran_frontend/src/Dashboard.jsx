import { useState } from "react";
import { searchVerses } from "./api";

function Dashboard({ token, onLogout }) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(50);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const RESULTS_PER_PAGE = 10;
  const [currentPage, setCurrentPage] = useState(1);

  const totalResults = results.length;
  const totalPages = Math.ceil(totalResults / RESULTS_PER_PAGE) || 1;
  const startIndex = (currentPage - 1) * RESULTS_PER_PAGE;
  const endIndex = startIndex + RESULTS_PER_PAGE;
  const currentPageResults = results.slice(startIndex, endIndex);

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

    setError("");
    setLoading(true);
    setResults([]);
    setCurrentPage(1);

    try {
      const data = await searchVerses(token, trimmed, topK);
      setResults(data);
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong while searching.");
    } finally {
      setLoading(false);
    }
  };

  const handleChangePage = (direction) => {
    setCurrentPage((prev) => {
      if (direction === "prev") return Math.max(1, prev - 1);
      return Math.min(totalPages, prev + 1);
    });
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

        {/* Intro box */}
        <div className="dashboard-intro">
          <p>
            This system provides <b>semantic search</b> for Qur&apos;anic verses.
          </p>
          <p>
            You can type a word in <b>Arabic</b> or <b>English</b>, and the
            system will retrieve the most related verses by <b>meaning</b>, not
            only by exact keywords.
          </p>
        </div>

        {/* Search form */}
        <form className="dashboard-search-form" onSubmit={handleSearch}>
          <input
            className="form-input"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your question..."
          />
          <select
            className="form-select"
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
          >
            <option value={5}>5 results</option>
            <option value={10}>10 results</option>
            <option value={20}>20 results</option>
            <option value={50}>50 results</option>
            <option value={100}>100 results</option>
          </select>
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

          {results.length > 0 && (
            <p className="results-summary">
              Found <b>{totalResults}</b> verses for query:{" "}
              <span className="results-query">{query}</span>
            </p>
          )}

          <div className="results-list">
            {currentPageResults.map((r) => (
              <div
                key={`${r.sura}:${r.ayah}-${r.rank}`}
                className="result-card"
              >
                <div className="result-meta">
                  <span>
                    Rank {r.rank} • Sura {r.sura}, Ayah {r.ayah}
                  </span>
                  <span>Score: {r.score.toFixed(3)}</span>
                </div>
                <p className="arabic-verse">{r.arabic}</p>
                <p className="english-verse">{r.english}</p>
              </div>
            ))}
          </div>

          {totalResults > 0 && (
            <div className="pagination-row">
              <span>
                Showing{" "}
                {totalResults === 0
                  ? 0
                  : `${startIndex + 1}–${Math.min(
                      endIndex,
                      totalResults
                    )}`}{" "}
                of {totalResults} results
              </span>

              {totalPages > 1 && (
                <div className="pagination-buttons">
                  <button
                    onClick={() => handleChangePage("prev")}
                    disabled={currentPage === 1}
                    className="pagination-btn"
                  >
                    Prev
                  </button>

                  <span>
                    Page {currentPage} of {totalPages}
                  </span>

                  <button
                    onClick={() => handleChangePage("next")}
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
