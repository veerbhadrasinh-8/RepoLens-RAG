// frontend/app.js
const API_BASE_URL = "http://localhost:4000/api";

/**
 * Handles user login form submission and persists JWT token.
 */
async function handleLogin(email, password) {
  try {
    const response = await fetch(`${API_BASE_URL}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      throw new Error("Login failed");
    }

    const data = await response.json();
    localStorage.setItem("authToken", data.token);
    window.location.href = "/dashboard.html";
  } catch (error) {
    alert("Authentication error: " + error.message);
  }
}

/**
 * Fetches user tasks using Bearer token from localStorage.
 */
async function loadTasks() {
  const token = localStorage.getItem("authToken");
  const response = await fetch(`${API_BASE_URL}/tasks`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return response.json();
}
