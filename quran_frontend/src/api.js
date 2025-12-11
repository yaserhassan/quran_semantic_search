import { API_BASE } from "./config";

export async function loginWithPassword(username, password, recaptchaToken) {
  const res = await fetch(`${API_BASE}/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      username: username.trim(),
      password: password.trim(),
      recaptcha_token: recaptchaToken,
    }),
  });

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new Error(data.detail || `Login failed (${res.status})`);
  }

  if (!data.access_token) {
    throw new Error("No access_token from login.");
  }

  return data.access_token;
}

export async function signup(username, password) {
  const res = await fetch(`${API_BASE}/signup`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      username: username.trim(),
      password: password.trim(),
    }),
  });

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new Error(data.detail || `Signup failed (${res.status})`);
  }

  return true;
}

export async function googleLogin(idToken) {
  const res = await fetch(`${API_BASE}/google-login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ id_token: idToken }),
  });

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new Error(data.detail || `Google login failed (${res.status})`);
  }

  if (!data.access_token) {
    throw new Error("No access_token from google-login.");
  }

  return data.access_token;
}

export async function searchVerses(token, query, topK) {
  const res = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      query: query.trim(),
      top_k: topK,
    }),
  });

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new Error(data.detail || `Search failed (${res.status})`);
  }

  return data.results || [];
}
