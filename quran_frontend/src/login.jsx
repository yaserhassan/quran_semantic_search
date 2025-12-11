import { useState } from "react";
import ReCAPTCHA from "react-google-recaptcha";
import { GoogleLogin } from "@react-oauth/google";
import { RECAPTCHA_SITE_KEY } from "./config";
import { loginWithPassword, signup, googleLogin } from "./api";

function Login({ onLoginSuccess }) {
  const [authMode, setAuthMode] = useState("login"); // login | signup

  // Login form
  const [loginUsername, setLoginUsername] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [loginLoading, setLoginLoading] = useState(false);
  const [loginError, setLoginError] = useState("");
  const [recaptchaToken, setRecaptchaToken] = useState("");

  // Signup form
  const [signupUsername, setSignupUsername] = useState("");
  const [signupPassword, setSignupPassword] = useState("");
  const [signupConfirm, setSignupConfirm] = useState("");
  const [signupLoading, setSignupLoading] = useState(false);
  const [signupError, setSignupError] = useState("");
  const [signupSuccess, setSignupSuccess] = useState("");

  // Google login handler
  const handleGoogleSuccess = async (credentialResponse) => {
    try {
      const idToken = credentialResponse?.credential;
      if (!idToken) {
        throw new Error("No Google ID token received.");
      }

      const accessToken = await googleLogin(idToken);
      onLoginSuccess(accessToken);
      setLoginError("");
      setSignupSuccess("");
    } catch (err) {
      console.error(err);
      setLoginError(err.message || "Google login failed. Please try again.");
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();

    if (!loginUsername.trim() || !loginPassword.trim()) {
      setLoginError("Please enter username and password.");
      return;
    }
    if (!recaptchaToken) {
      setLoginError("Please complete the reCAPTCHA.");
      return;
    }

    setLoginError("");
    setSignupSuccess("");
    setLoginLoading(true);

    try {
      const accessToken = await loginWithPassword(
        loginUsername,
        loginPassword,
        recaptchaToken
      );
      onLoginSuccess(accessToken);
    } catch (err) {
      console.error(err);
      setLoginError(err.message || "Login failed. Please try again.");
    } finally {
      setLoginLoading(false);
    }
  };

  const handleSignup = async (e) => {
    e.preventDefault();

    const u = signupUsername.trim();
    const p = signupPassword.trim();
    const c = signupConfirm.trim();

    if (!u || !p || !c) {
      setSignupError("All fields are required.");
      return;
    }
    if (p.length < 6) {
      setSignupError("Password must be at least 6 characters.");
      return;
    }
    if (p !== c) {
      setSignupError("Passwords do not match.");
      return;
    }

    setSignupError("");
    setSignupSuccess("");
    setSignupLoading(true);

    try {
      await signup(u, p);
      setSignupSuccess("Account created successfully. You can now log in.");
      setLoginUsername(u);
      setAuthMode("login");
      setSignupPassword("");
      setSignupConfirm("");
    } catch (err) {
      console.error(err);
      setSignupError(err.message || "Signup failed. Please try again.");
    } finally {
      setSignupLoading(false);
    }
  };

  return (
    <div className="page-bg">
      <div className="login-shell">
        {/* Right info panel */}
        <div className="login-info-panel">
          <div>
            <div className="login-pill">Qur&apos;an Semantic Search</div>

            <h2 className="login-title">
              Discover Qur&apos;anic verses by meaning
            </h2>

            <p className="login-text">
              This project provides a <strong>semantic search engine</strong>{" "}
              over the Qur&apos;an. Instead of only matching exact words, the
              system understands <strong>concepts</strong> in Arabic and
              English, and retrieves the most related verses.
            </p>

            <ul className="login-list">
              <li>Search using Arabic terms such as: البصر، السمع، الصبر.</li>
              <li>
                Or English concepts like: &quot;patience&quot;, &quot;hearing&quot;, &quot;Judgment Day&quot;.
              </li>
              <li>Built with BGE-M3 embeddings and MAQAS morphological data.</li>
            </ul>
          </div>
        </div>

        {/* Left auth panel */}
        <div className="login-auth-panel">
          <div className="login-auth-header">
            <h2 className="login-auth-title">
              {authMode === "login" ? "Welcome back" : "Create your account"}
            </h2>
            <p className="login-auth-subtitle">
              {authMode === "login"
                ? "Sign in to access the Qur'an semantic search dashboard."
                : "Register to explore Qur'anic verses through semantic search."}
            </p>
          </div>

          <GoogleLogin
            onSuccess={handleGoogleSuccess}
            onError={() =>
              setLoginError("Google login was cancelled or failed.")
            }
            width="100%"
            shape="pill"
          />

          <div className="login-divider">
            <div className="login-divider-line" />
            <span className="login-divider-text">or</span>
            <div className="login-divider-line" />
          </div>

          {authMode === "login" ? (
            <form className="login-form" onSubmit={handleLogin}>
              <div className="form-field">
                <label className="form-label">Username</label>
                <input
                  className="form-input"
                  type="text"
                  value={loginUsername}
                  onChange={(e) => setLoginUsername(e.target.value)}
                />
              </div>

              <div className="form-field">
                <label className="form-label">Password</label>
                <input
                  className="form-input"
                  type="password"
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                />
              </div>

              <div className="forgot-row">
                <span className="forgot-text">Forgot password?</span>
              </div>

              <div className="recaptcha-wrapper">
                <ReCAPTCHA
                  sitekey={RECAPTCHA_SITE_KEY}
                  onChange={(value) => setRecaptchaToken(value || "")}
                />
              </div>

              {loginError && <div className="alert-error">{loginError}</div>}

              {signupSuccess && (
                <div className="alert-success">{signupSuccess}</div>
              )}

              <button
                type="submit"
                disabled={loginLoading}
                className="btn-primary full-width"
              >
                {loginLoading ? "Signing in..." : "Login"}
              </button>

              <div className="auth-switch">
                Don&apos;t have an account?{" "}
                <button
                  type="button"
                  className="link-button"
                  onClick={() => {
                    setAuthMode("signup");
                    setLoginError("");
                  }}
                >
                  Create account
                </button>
              </div>
            </form>
          ) : (
            <form className="login-form" onSubmit={handleSignup}>
              <div className="form-field">
                <label className="form-label">Username</label>
                <input
                  className="form-input"
                  type="text"
                  value={signupUsername}
                  onChange={(e) => setSignupUsername(e.target.value)}
                />
              </div>

              <div className="form-field">
                <label className="form-label">Password</label>
                <input
                  className="form-input"
                  type="password"
                  value={signupPassword}
                  onChange={(e) => setSignupPassword(e.target.value)}
                />
              </div>

              <div className="form-field">
                <label className="form-label">Confirm password</label>
                <input
                  className="form-input"
                  type="password"
                  value={signupConfirm}
                  onChange={(e) => setSignupConfirm(e.target.value)}
                />
              </div>

              {signupError && (
                <div className="alert-error">{signupError}</div>
              )}

              <button
                type="submit"
                disabled={signupLoading}
                className="btn-primary full-width"
              >
                {signupLoading ? "Creating account..." : "Create account"}
              </button>

              <div className="auth-switch">
                Already have an account?{" "}
                <button
                  type="button"
                  className="link-button"
                  onClick={() => {
                    setAuthMode("login");
                    setSignupError("");
                  }}
                >
                  Log in
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}

export default Login;
